/*
 * Copyright (c) 2022, RISE Research Institutes of Sweden AB.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "contiki.h"
#include "lib/random.h"
#include "net/routing/routing.h"
#include "net/netstack.h"
#include "net/ipv6/simple-udp.h"
#include "sys/energest.h"
#include "app-message.h"
//#include "icmp6-stats.h"
#include "network-attacks.h"
#include "net/ipv6/uiplib.h" //EDIT
//#include "rpl_of0_worst.h"

#if ROUTING_CONF_RPL_LITE
#include "net/routing/rpl-lite/rpl.h"
#include "net/routing/rpl-lite/rpl-timers.h"
#elif ROUTING_CONF_RPL_CLASSIC
#include "net/routing/rpl-classic/rpl.h"
#include "net/routing/rpl-classic/rpl-private.h"
#endif
#include <inttypes.h>

#include "sys/log.h"
#define LOG_MODULE "App"
#define LOG_LEVEL LOG_LEVEL_INFO

#define WITH_SERVER_REPLY  1
#define UDP_CLIENT_PORT	8765
#define UDP_SERVER_PORT	5678
#define NODE_TO_NODE_PORT 8766  // New port for node-to-node communication

#define SEND_INTERVAL		  (60 * CLOCK_SECOND)
#define NORMAL_TRAFFIC_MAX_INTERVAL (180 * CLOCK_SECOND) //EDIT: Maximum delay between packets
#define SHOW_ENERGEST             (0 && ENERGEST_CONF_ON)

static struct simple_udp_connection udp_conn;
static struct simple_udp_connection node_to_node_conn;

/*---------------------------------------------------------------------------*/
PROCESS(udp_client_process, "UDP client");
PROCESS(normal_udp_traffic_process, "Normal UDP traffic"); //EDIT
AUTOSTART_PROCESSES(&udp_client_process, &normal_udp_traffic_process); //EDIT
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
  LOG_INFO("Received ");
  if(datalen != sizeof(app_message_t)) {
    LOG_INFO_("unknown data of size %u from ", datalen);
  } else {
    LOG_INFO_("response %"PRIu32" from ", app_read_uint32(msg->seqno));
  }
  LOG_INFO_6ADDR(sender_addr);
#if LLSEC802154_CONF_ENABLED
  LOG_INFO_(" LLSEC LV:%d", uipbuf_get_attr(UIPBUF_ATTR_LLSEC_LEVEL));
#endif
  LOG_INFO_("\n");
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
  uint8_t hop_limit = UIP_IP_BUF->ttl; // Get remaining hop limit from IPv6 header

  LOG_INFO("HOPCOUNTMSG ");
  if(datalen != sizeof(int)) {
    LOG_INFO_("unknown data of size %u from ", datalen);
  } else {
    LOG_INFO_("Data: %u", *msg); // Print received uint8_t data
  }
  LOG_INFO_(" from ");
  LOG_INFO_6ADDR(sender_addr);

  LOG_INFO_(", Hop Count: %u", (64 - hop_limit));

#if LLSEC802154_CONF_ENABLED
  LOG_INFO_(", LLSEC LV:%d", uipbuf_get_attr(UIPBUF_ATTR_LLSEC_LEVEL));
#endif
  LOG_INFO_("\n");
}

/*---------------------------------------------------------------------------*/

// List of known nodes (hardcoded for simplicity)
static char *node_ip_list[] = {
  "fd00::201:1:1:1", 
  "fd00::202:2:2:2",
  "fd00::203:3:3:3",
  "fd00::204:4:4:4",
  "fd00::205:5:5:5",
  "fd00::206:6:6:6",
  "fd00::207:7:7:7",
  "fd00::208:8:8:8",
  "fd00::209:9:9:9",
  "fd00::20a:a:a:a",
  "fd00::20b:b:b:b",
  "fd00::20c:c:c:c",
  "fd00::20d:d:d:d",
  "fd00::20e:e:e:e",
  "fd00::20f:f:f:f",
  "fd00::210:10:10:10",
  "fd00::211:11:11:11",
  "fd00::212:12:12:12",
  "fd00::213:13:13:13",
  "fd00::214:14:14:14",
  "fd00::215:15:15:15"
};

// Function to get a random IP address
static char* get_random_node_ip(uip_ipaddr_t *dest_ipaddr) {
  //int random_index = random_rand() % num_of_motes_including_sink;
  uiplib_ipaddrconv(node_ip_list[0], dest_ipaddr); //random index
  return node_ip_list[0]; //random index
}
/*---------------------------------------------------------------------------*/
#if SHOW_ENERGEST
static inline unsigned long
to_seconds(uint64_t time)
{
  return (unsigned long)(time / ENERGEST_SECOND);
}
#endif /* SHOW_ENERGEST */
/*---------------------------------------------------------------------------*/
PROCESS_THREAD(udp_client_process, ev, data)
{
  static struct etimer periodic_timer;
  static uint32_t count;
  static app_message_t msg;
  uip_ipaddr_t dest_ipaddr;
  rpl_instance_t *default_instance;
  uint16_t rank;
  uint8_t dag_version;

  PROCESS_BEGIN();

  /* Initialize network attacks support */
  network_attacks_init();


  /* Initialize UDP connection */
  simple_udp_register(&udp_conn, UDP_CLIENT_PORT, NULL,
                      UDP_SERVER_PORT, udp_rx_callback);

  etimer_set(&periodic_timer, random_rand() % SEND_INTERVAL);
  while(1) {
    PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&periodic_timer));

    if(NETSTACK_ROUTING.node_is_reachable() && NETSTACK_ROUTING.get_root_ipaddr(&dest_ipaddr)) {
      /* Send to DAG root */
      LOG_INFO("Sending request %"PRIu32" to ", count);
      LOG_INFO_6ADDR(&dest_ipaddr);
      LOG_INFO_("\n");
      memset(&msg, 0, sizeof(msg));
      app_write_uint32(msg.seqno, count);
      default_instance = rpl_get_default_instance();
      rank = default_instance ? default_instance->dag.rank : RPL_INFINITE_RANK;
      dag_version = default_instance ? default_instance->dag.version : 0;
      app_write_uint16(msg.rpl_rank, rank);
/* */
     LOG_INFO("DATA: sq:%" PRIu32 ",rank:%3u,ver:%u", count, rank, dag_version);
     LOG_INFO_(",disr:%"PRIu32",diss:%"PRIu32,
              icmp6_stats.dis_uc_recv + icmp6_stats.dis_mc_recv,
              icmp6_stats.dis_uc_sent + icmp6_stats.dis_mc_sent);

     LOG_INFO_(",dior:%"PRIu32",dios:%"PRIu32, 
              icmp6_stats.dio_uc_recv + icmp6_stats.dio_mc_recv,
              icmp6_stats.dio_uc_sent + icmp6_stats.dio_mc_sent);
     LOG_INFO_(",diar:%"PRIu32",tots:%"PRIu32"\n", 
              icmp6_stats.dao_recv, icmp6_stats.rpl_total_sent);

/* */    
      msg.rpl_dag_version = dag_version;
      simple_udp_sendto(&udp_conn, &msg, sizeof(msg), &dest_ipaddr);
      count++;
    } else {
      LOG_INFO("Not reachable yet\n");
    }

    /* Add some jitter */
    etimer_set(&periodic_timer, SEND_INTERVAL
      - CLOCK_SECOND + (random_rand() % (2 * CLOCK_SECOND)));

    /* Example printing energest data */
#if SHOW_ENERGEST
    /*
     * Update all energest times. Should always be called before energest
     * times are read.
     */
    energest_flush();

    printf("\nEnergest:\n");
    printf(" CPU          %4lus LPM      %4lus DEEP LPM %4lus  Total time %lus\n",
           to_seconds(energest_type_time(ENERGEST_TYPE_CPU)),
           to_seconds(energest_type_time(ENERGEST_TYPE_LPM)),
           to_seconds(energest_type_time(ENERGEST_TYPE_DEEP_LPM)),
           to_seconds(ENERGEST_GET_TOTAL_TIME()));
    printf(" Radio LISTEN %4lus TRANSMIT %4lus OFF      %4lus\n",
           to_seconds(energest_type_time(ENERGEST_TYPE_LISTEN)),
           to_seconds(energest_type_time(ENERGEST_TYPE_TRANSMIT)),
           to_seconds(ENERGEST_GET_TOTAL_TIME()
                      - energest_type_time(ENERGEST_TYPE_TRANSMIT)
                      - energest_type_time(ENERGEST_TYPE_LISTEN)));
#endif /* SHOW_ENERGEST */
  }

  PROCESS_END();
}
/*---------------------------------------------------------------------------*/
/* Normal Traffic Simulation Process */
//ALL BELOW IS EDIT
PROCESS_THREAD(normal_udp_traffic_process, ev, data)
{
  static struct etimer normal_traffic_timer;
  int payload = get_of_counter();
  //uint8_t payload = 1;
  uip_ipaddr_t dest_ip;
  char* str_addr = NULL;

  PROCESS_BEGIN();

  simple_udp_register(&node_to_node_conn, NODE_TO_NODE_PORT, NULL,
    NODE_TO_NODE_PORT, udp_node_to_node_callback);

  etimer_set(&normal_traffic_timer, random_rand() % NORMAL_TRAFFIC_MAX_INTERVAL);

  while(1) {
    PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&normal_traffic_timer));

    str_addr = get_random_node_ip(&dest_ip);
    payload = get_of_counter();
      if(str_addr != NULL) {
        LOG_INFO("Sending HOPCOUNTMSG message to sink\n");
        //LOG_INFO_("\n");

        simple_udp_sendto(&node_to_node_conn, &payload, sizeof(int), &dest_ip);
      }
      else {
      LOG_INFO("No known routes, skipping normal traffic\n");
    }

    uint16_t next_interval = (random_rand() % NORMAL_TRAFFIC_MAX_INTERVAL) + CLOCK_SECOND;
    etimer_set(&normal_traffic_timer, next_interval);
  }

  PROCESS_END();
}