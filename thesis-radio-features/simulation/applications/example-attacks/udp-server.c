#include "contiki.h"
#include "net/routing/routing.h"
#include "net/netstack.h"
#include "net/ipv6/simple-udp.h"
#include "net/packetbuf.h"
#include "app-message.h"
#include "network-attacks.h"
#include <inttypes.h>

#include "sys/log.h"
#define LOG_MODULE "App"
#define LOG_LEVEL LOG_LEVEL_INFO

#define WITH_SERVER_REPLY  1
#define UDP_CLIENT_PORT 8765
#define UDP_SERVER_PORT 5678
#define NODE_TO_NODE_PORT 8766

static struct simple_udp_connection udp_conn;
static struct simple_udp_connection node_to_node_conn;

PROCESS(udp_server_process, "UDP server");
AUTOSTART_PROCESSES(&udp_server_process);
/*---------------------------------------------------------------------------*/
static void
udp_rx_callback(struct simple_udp_connection *c,
         const uip_ipaddr_t *sender_addr,
         uint16_t sender_port,
         const uip_ipaddr_t *receiver_addr,
         uint16_t receiver_port,
         const uint8_t *data,
         uint16_t datalen)
{
  app_message_t *msg = (app_message_t *)data;
  int16_t rssi = (int16_t)packetbuf_attr(PACKETBUF_ATTR_RSSI);
  int lqi = packetbuf_attr(PACKETBUF_ATTR_LINK_QUALITY);

  LOG_INFO("Received ");
  if(datalen != sizeof(app_message_t)) {
    LOG_INFO_("unknown data of size %u from ", datalen);
  } else {
    LOG_INFO_(" message %"PRIu32" (rank %u, rssi %d, lqi %d) from ",
              app_read_uint32(msg->seqno),
              app_read_uint16(msg->rpl_rank),
              rssi,
              lqi);
  }
  LOG_INFO_6ADDR(sender_addr);
  LOG_INFO_("\n");

#if WITH_SERVER_REPLY
  LOG_INFO("Sending response.\n");
  simple_udp_sendto(&udp_conn, data, datalen, sender_addr);
#endif
}

static void
udp_node_to_node_callback(struct simple_udp_connection *c,
         const uip_ipaddr_t *sender_addr,
         uint16_t sender_port,
         const uip_ipaddr_t *receiver_addr,
         uint16_t receiver_port,
         const uint8_t *data,
         uint16_t datalen)
{
  int *msg = (int *)data;
  uint8_t hop_limit = UIP_IP_BUF->ttl;
  int16_t rssi = (int16_t)packetbuf_attr(PACKETBUF_ATTR_RSSI);
  int lqi = packetbuf_attr(PACKETBUF_ATTR_LINK_QUALITY);

  LOG_INFO("HOPCOUNTMSG ");
  if(datalen != sizeof(int)) {
    LOG_INFO_("unknown data of size %u from ", datalen);
  } else {
    LOG_INFO_("Data: %u", *msg);
  }
  LOG_INFO_(" from ");
  LOG_INFO_6ADDR(sender_addr);
  LOG_INFO_(", Hop Count: %u, RSSI: %d, LQI: %d", (64 - hop_limit), rssi, lqi);

#if LLSEC802154_CONF_ENABLED
  LOG_INFO_(", LLSEC LV:%d", uipbuf_get_attr(UIPBUF_ATTR_LLSEC_LEVEL));
#endif
  LOG_INFO_("\n");
}
/*---------------------------------------------------------------------------*/
PROCESS_THREAD(udp_server_process, ev, data)
{
  PROCESS_BEGIN();

  network_attacks_init();
  NETSTACK_ROUTING.root_start();

  simple_udp_register(&udp_conn, UDP_SERVER_PORT, NULL,
                      UDP_CLIENT_PORT, udp_rx_callback);

  simple_udp_register(&node_to_node_conn, NODE_TO_NODE_PORT, NULL,
                       NODE_TO_NODE_PORT, udp_node_to_node_callback);

  PROCESS_END();
}
