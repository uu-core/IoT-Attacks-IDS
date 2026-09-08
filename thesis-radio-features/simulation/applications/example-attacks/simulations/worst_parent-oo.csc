<?xml version="1.0" encoding="UTF-8"?>
<simconf>
  <simulation>
    <title>My simulation</title>
    <randomseed>123456</randomseed>
    <motedelay_us>1000000</motedelay_us>
    <radiomedium>
      org.contikios.cooja.radiomediums.UDGM
      <transmitting_range>50.0</transmitting_range>
      <interference_range>100.0</interference_range>
      <success_ratio_tx>1.0</success_ratio_tx>
      <success_ratio_rx>1.0</success_ratio_rx>
    </radiomedium>
    <events>
      <logoutput>40000</logoutput>
      <datatrace>true</datatrace>
    </events>
    <motetype>
      org.contikios.cooja.contikimote.ContikiMoteType
      <identifier>mtype519721567</identifier>
      <description>Cooja Mote Type #1</description>
      <source>[CONFIG_DIR]/../../../../udp-server.c</source>
      <commands>make -j$(CPUS) udp-server.cooja TARGET=cooja</commands>
      <moteinterface>org.contikios.cooja.interfaces.Position</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.Battery</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiVib</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiMoteID</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiRS232</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiBeeper</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.RimeAddress</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiIPAddress</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiRadio</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiButton</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiPIR</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiClock</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiLED</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiCFS</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiEEPROM</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.Mote2MoteRelations</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.MoteAttributes</moteinterface>
    </motetype>
    <motetype>
      org.contikios.cooja.contikimote.ContikiMoteType
      <identifier>mtype603107969</identifier>
      <description>Cooja Mote Type #2</description>
      <source>[CONFIG_DIR]/../../../../udp-client.c</source>
      <commands>make -j$(CPUS) udp-client.cooja TARGET=cooja</commands>
      <moteinterface>org.contikios.cooja.interfaces.Position</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.Battery</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiVib</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiMoteID</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiRS232</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiBeeper</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.RimeAddress</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiIPAddress</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiRadio</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiButton</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiPIR</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiClock</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiLED</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiCFS</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiEEPROM</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.Mote2MoteRelations</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.MoteAttributes</moteinterface>
    </motetype>
    <mote>
      <interface_config>
        org.contikios.cooja.interfaces.Position
        <x>50.00</x>
        <y>50.00</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
        org.contikios.cooja.contikimote.interfaces.ContikiMoteID
        <id>1</id>
      </interface_config>
      <motetype_identifier>mtype519721567</motetype_identifier>
    </mote>
  </simulation>
  <plugin>
    org.contikios.cooja.plugins.SimControl
    <width>280</width>
    <z>0</z>
    <height>160</height>
    <location_x>220</location_x>
    <location_y>189</location_y>
  </plugin>
  <plugin>
    org.contikios.cooja.plugins.Visualizer
    <plugin_config>
      <moterelations>true</moterelations>
      <skin>org.contikios.cooja.plugins.skins.IDVisualizerSkin</skin>
      <skin>org.contikios.cooja.plugins.skins.GridVisualizerSkin</skin>
      <skin>org.contikios.cooja.plugins.skins.TrafficVisualizerSkin</skin>
      <skin>org.contikios.cooja.plugins.skins.UDGMVisualizerSkin</skin>
      <viewport>5.67917169874098 0.0 0.0 5.67917169874098 -231.6681334490396 -109.37283490320507</viewport>
    </plugin_config>
    <width>400</width>
    <z>4</z>
    <height>400</height>
    <location_x>1</location_x>
    <location_y>1</location_y>
  </plugin>
  <plugin>
    org.contikios.cooja.plugins.Notes
    <plugin_config>
      <notes>Enter notes here</notes>
      <decorations>true</decorations>
    </plugin_config>
    <width>150</width>
    <z>2</z>
    <height>300</height>
    <location_x>680</location_x>
    <location_y>0</location_y>
  </plugin>
  <plugin>
    org.contikios.cooja.plugins.LogListener
    <plugin_config>
      <filter>App</filter>
      <formatted_time />
      <coloring />
    </plugin_config>
    <width>893</width>
    <z>2</z>
    <height>470</height>
    <location_x>0</location_x>
    <location_y>326</location_y>
  </plugin>
<plugin>
    org.contikios.cooja.plugins.ScriptRunner
    <plugin_config>
  <script>
var senders = {};
var verbose = false;
var waiting_for_stable_network = true;
var sinkId = 1;
// Number of clients (the sink excluded)
var clients = sim.getMotesCount() - 1;
var msgrecv = /.+INFO: App.+Received +message.+ from ([0-9a-f:]+).*/;
var r = new java.util.Random(sim.getRandomSeed());

/* timeout in milliseconds */
TIMEOUT(65000000);

function f(value) {
  return (Math.round(value * 100) / 100).toFixed(2);
}

function setBool(mote, name, value) {
  var mem = new org.contikios.cooja.mote.memory.VarMemory(mote.getMemory());
  if (!mem.variableExists(name)) {
    log.log("ERR: could not find variable '" + name + "'\n");
    return false;
  }
  var symbol = mem.getVariable(name);
  if (verbose) {
    var oldValue = mem.getInt8ValueOf(symbol.addr) ? "true" : "false";
    log.log("Set bool " + name + " (address 0x" + java.lang.Long.toHexString(symbol.addr)
            + "/" + symbol.size + ": " + oldValue + ") to " + value + "\n");
  }
  mem.setInt8ValueOf(symbol.addr, value);
  return true;
}

function setInt16(mote, name, value) {
  var mem = new org.contikios.cooja.mote.memory.VarMemory(mote.getMemory());
  if (!mem.variableExists(name)) {
    log.log("ERR: could not find variable '" + name + "'\n");
    return false;
  }
  var symbol = mem.getVariable(name);
  if (verbose) {
    var oldValue = mem.getInt16ValueOf(symbol.addr) &amp; 0xffff;
    log.log("Set int16 " + name + " (address 0x" + java.lang.Long.toHexString(symbol.addr)
            + "/" + symbol.size + ": " + oldValue + ") to " + value + "\n");
  }
  mem.setInt16ValueOf(symbol.addr, value);
  return true;
}

for(var wthid = 1; wthid &lt; clients + 2; wthid++) { /*assumes sink has id 1*/
  setInt16(sim.getMoteWithID(wthid), 'num_of_motes_including_sink', clients + 1);
}

function selectAttacker() {
  var sink = sim.getMoteWithID(sinkId);
  var sinkRadio = sink.getInterfaces().getRadio();
  while (true) {
    var attackerId = 2 + r.nextInt(clients);
    log.log("Checking potential attacker " + attackerId + "... ");

    var attacker = sim.getMoteWithID(attackerId);
    var attackerRadio = attacker.getInterfaces().getRadio();
    var neighbours = sim.getRadioMedium().getNeighbours(attackerRadio);
    if (neighbours.contains(sinkRadio)) {
      log.log("[FAIL] - has sink as neighbour\n");
      continue;
    }
    log.log("[OK]\n");
    return attacker;}}

while(waiting_for_stable_network) {
    YIELD();
    if (id == 1) {
        match = msg.match(msgrecv)
        if (match) {
            senders[match[1]] = true;
            var size = Object.keys(senders).length;
            log.log("Sink has contact with " + match[1] + " (" + (clients - size) + " remaining)\n");
            if (size &gt;= clients) {
                log.log("Sink has contact with all clients!\n");
                waiting_for_stable_network = false;
            }
        }
    }
}

GENERATE_MSG(2000, "continue");
YIELD_THEN_WAIT_UNTIL(msg.equals("continue"));

sim.getEventCentral().logEvent("network", "steady-state");
log.log("Network steady state!\n");


var attacker = selectAttacker(); 

// Start Attack
GENERATE_MSG(27000000, "continue");
YIELD_THEN_WAIT_UNTIL(msg.equals("continue"));


for(var i = 0; i &lt; 15; i++) { // 14 times, change if need
  // Start Attack
  GENERATE_MSG(900000, "continue");
  YIELD_THEN_WAIT_UNTIL(msg.equals("continue"));

  log.log("Worst Parent attack from " + attacker.getID() + "!\n");
  if (i == 0) {
    sim.getEventCentral().logEvent("attack", "wpa:" + attacker.getID());
  }

  /* Configure WPA random attack */
  setBool(attacker, 'network_attacks_worst_parent', true);
  setInt16(attacker, 'network_attacks_rpl_dio_fake_rank', 128);
  setBool(attacker, 'network_attacks_rpl_dio_reset', true);


  // Stop Attack after 15 mins
  GENERATE_MSG(900000, "stop-attack");
  YIELD_THEN_WAIT_UNTIL(msg.equals("stop-attack"));

  log.log("Stopped - Worst Parent attack from " + attacker.getID() + "!\n");
  //sim.getEventCentral().logEvent("stop-attack", "wpa:" + attacker.getID());

  setInt16(attacker, 'network_attacks_rpl_dio_fake_rank', 0);
  setBool(attacker, 'network_attacks_rpl_dio_reset', true);
}

success = true;

log.testOK();

</script>
      <active>true</active>
    </plugin_config>
    <width>696</width>
    <z>0</z>
    <height>642</height>
    <location_x>718</location_x>
    <location_y>73</location_y>
  </plugin>
</simconf>

