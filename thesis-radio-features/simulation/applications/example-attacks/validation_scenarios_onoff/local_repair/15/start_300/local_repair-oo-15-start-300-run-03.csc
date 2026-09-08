<?xml version='1.0' encoding='utf-8'?>
<simconf>
  <simulation>
    <title>Controlled on-off validation: local_repair, 15 nodes, start 300 min, run 3</title>
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
      <source>[CONFIG_DIR]/../../../../udp-client-validation.c</source>
      <commands>make -j$(CPUS) udp-client-validation.cooja TARGET=cooja</commands>
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
    <mote>
      <interface_config>
 org.contikios.cooja.interfaces.Position
 <x>46</x>
        <y>82</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
 org.contikios.cooja.contikimote.interfaces.ContikiMoteID
 <id>2</id>
      </interface_config>
      <motetype_identifier>mtype603107969</motetype_identifier>
    </mote>
    <mote>
      <interface_config>
 org.contikios.cooja.interfaces.Position
 <x>82</x>
        <y>30</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
 org.contikios.cooja.contikimote.interfaces.ContikiMoteID
 <id>3</id>
      </interface_config>
      <motetype_identifier>mtype603107969</motetype_identifier>
    </mote>
    <mote>
      <interface_config>
 org.contikios.cooja.interfaces.Position
 <x>71</x>
        <y>6</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
 org.contikios.cooja.contikimote.interfaces.ContikiMoteID
 <id>4</id>
      </interface_config>
      <motetype_identifier>mtype603107969</motetype_identifier>
    </mote>
    <mote>
      <interface_config>
 org.contikios.cooja.interfaces.Position
 <x>108</x>
        <y>22</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
 org.contikios.cooja.contikimote.interfaces.ContikiMoteID
 <id>5</id>
      </interface_config>
      <motetype_identifier>mtype603107969</motetype_identifier>
    </mote>
    <mote>
      <interface_config>
 org.contikios.cooja.interfaces.Position
 <x>46</x>
        <y>115</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
 org.contikios.cooja.contikimote.interfaces.ContikiMoteID
 <id>6</id>
      </interface_config>
      <motetype_identifier>mtype603107969</motetype_identifier>
    </mote>
    <mote>
      <interface_config>
 org.contikios.cooja.interfaces.Position
 <x>129</x>
        <y>67</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
 org.contikios.cooja.contikimote.interfaces.ContikiMoteID
 <id>7</id>
      </interface_config>
      <motetype_identifier>mtype603107969</motetype_identifier>
    </mote>
    <mote>
      <interface_config>
 org.contikios.cooja.interfaces.Position
 <x>23</x>
        <y>86</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
 org.contikios.cooja.contikimote.interfaces.ContikiMoteID
 <id>8</id>
      </interface_config>
      <motetype_identifier>mtype603107969</motetype_identifier>
    </mote>
    <mote>
      <interface_config>
 org.contikios.cooja.interfaces.Position
 <x>164</x>
        <y>64</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
 org.contikios.cooja.contikimote.interfaces.ContikiMoteID
 <id>9</id>
      </interface_config>
      <motetype_identifier>mtype603107969</motetype_identifier>
    </mote>
    <mote>
      <interface_config>
 org.contikios.cooja.interfaces.Position
 <x>107</x>
        <y>94</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
 org.contikios.cooja.contikimote.interfaces.ContikiMoteID
 <id>10</id>
      </interface_config>
      <motetype_identifier>mtype603107969</motetype_identifier>
    </mote>
    <mote>
      <interface_config>
 org.contikios.cooja.interfaces.Position
 <x>118</x>
        <y>135</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
 org.contikios.cooja.contikimote.interfaces.ContikiMoteID
 <id>11</id>
      </interface_config>
      <motetype_identifier>mtype603107969</motetype_identifier>
    </mote>
    <mote>
      <interface_config>
 org.contikios.cooja.interfaces.Position
 <x>167</x>
        <y>100</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
 org.contikios.cooja.contikimote.interfaces.ContikiMoteID
 <id>12</id>
      </interface_config>
      <motetype_identifier>mtype603107969</motetype_identifier>
    </mote>
    <mote>
      <interface_config>
 org.contikios.cooja.interfaces.Position
 <x>16</x>
        <y>150</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
 org.contikios.cooja.contikimote.interfaces.ContikiMoteID
 <id>13</id>
      </interface_config>
      <motetype_identifier>mtype603107969</motetype_identifier>
    </mote>
    <mote>
      <interface_config>
 org.contikios.cooja.interfaces.Position
 <x>177</x>
        <y>25</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
 org.contikios.cooja.contikimote.interfaces.ContikiMoteID
 <id>14</id>
      </interface_config>
      <motetype_identifier>mtype603107969</motetype_identifier>
    </mote>
    <mote>
      <interface_config>
 org.contikios.cooja.interfaces.Position
 <x>33</x>
        <y>104</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
 org.contikios.cooja.contikimote.interfaces.ContikiMoteID
 <id>15</id>
      </interface_config>
      <motetype_identifier>mtype603107969</motetype_identifier>
    </mote>
    <mote>
      <interface_config>
 org.contikios.cooja.interfaces.Position
 <x>60</x>
        <y>168</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
 org.contikios.cooja.contikimote.interfaces.ContikiMoteID
 <id>16</id>
      </interface_config>
      <motetype_identifier>mtype603107969</motetype_identifier>
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
  return sim.getMoteWithID(3);
}
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

for(var wthid = 2; wthid &lt; clients + 2; wthid++) { /*assumes sink has id 1*/
  setBool(sim.getMoteWithID(wthid), 'network_attacks_set_of_to_counter', true);
}


GENERATE_MSG(5000, "continue");
YIELD_THEN_WAIT_UNTIL(msg.equals("continue"));

var attacker = selectAttacker();


/* Wait until the controlled attack-start time */
GENERATE_MSG(17995000, "attack_start");
YIELD_THEN_WAIT_UNTIL(msg.equals("attack_start"));

log.log("Local Repair attack ON from " + attacker.getID() + "!\n");

sim.getEventCentral().logEvent(
  "attack",
  "localrepair:" + attacker.getID()
);


/* Cycle 1: ON */

GENERATE_MSG(900000, "on_done_1");
GENERATE_MSG(30000, "dio_1");

while (true) {
  YIELD();

  if (msg.equals("dio_1")) {

    setBool(
      attacker,
      'network_attacks_local_repair_dio_send',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_reset',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_send',
      true
    );

    GENERATE_MSG(
      30000,
      "dio_1"
    );

  } else if (msg.equals("on_done_1")) {
    break;
  }
}

log.log(
  "Local Repair attack OFF from "
  + attacker.getID()
  + "!\n"
);

sim.getEventCentral().logEvent(
  "stop-attack",
  "localrepair:" + attacker.getID()
);

setBool(
  attacker,
  'network_attacks_rpl_dio_send',
  true
);

setBool(
  attacker,
  'network_attacks_rpl_dio_reset',
  true
);


/* Cycle 1: OFF */

GENERATE_MSG(900000, "off_done_1");

while (true) {
  YIELD();

  if (msg.equals("off_done_1")) {
    break;
  }
}

log.log(
  "Local Repair attack ON from "
  + attacker.getID()
  + "!\n"
);


/* Cycle 2: ON */

GENERATE_MSG(900000, "on_done_2");
GENERATE_MSG(30000, "dio_2");

while (true) {
  YIELD();

  if (msg.equals("dio_2")) {

    setBool(
      attacker,
      'network_attacks_local_repair_dio_send',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_reset',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_send',
      true
    );

    GENERATE_MSG(
      30000,
      "dio_2"
    );

  } else if (msg.equals("on_done_2")) {
    break;
  }
}

log.log(
  "Local Repair attack OFF from "
  + attacker.getID()
  + "!\n"
);

sim.getEventCentral().logEvent(
  "stop-attack",
  "localrepair:" + attacker.getID()
);

setBool(
  attacker,
  'network_attacks_rpl_dio_send',
  true
);

setBool(
  attacker,
  'network_attacks_rpl_dio_reset',
  true
);


/* Cycle 2: OFF */

GENERATE_MSG(900000, "off_done_2");

while (true) {
  YIELD();

  if (msg.equals("off_done_2")) {
    break;
  }
}

log.log(
  "Local Repair attack ON from "
  + attacker.getID()
  + "!\n"
);


/* Cycle 3: ON */

GENERATE_MSG(900000, "on_done_3");
GENERATE_MSG(30000, "dio_3");

while (true) {
  YIELD();

  if (msg.equals("dio_3")) {

    setBool(
      attacker,
      'network_attacks_local_repair_dio_send',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_reset',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_send',
      true
    );

    GENERATE_MSG(
      30000,
      "dio_3"
    );

  } else if (msg.equals("on_done_3")) {
    break;
  }
}

log.log(
  "Local Repair attack OFF from "
  + attacker.getID()
  + "!\n"
);

sim.getEventCentral().logEvent(
  "stop-attack",
  "localrepair:" + attacker.getID()
);

setBool(
  attacker,
  'network_attacks_rpl_dio_send',
  true
);

setBool(
  attacker,
  'network_attacks_rpl_dio_reset',
  true
);


/* Cycle 3: OFF */

GENERATE_MSG(900000, "off_done_3");

while (true) {
  YIELD();

  if (msg.equals("off_done_3")) {
    break;
  }
}

log.log(
  "Local Repair attack ON from "
  + attacker.getID()
  + "!\n"
);


/* Cycle 4: ON */

GENERATE_MSG(900000, "on_done_4");
GENERATE_MSG(30000, "dio_4");

while (true) {
  YIELD();

  if (msg.equals("dio_4")) {

    setBool(
      attacker,
      'network_attacks_local_repair_dio_send',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_reset',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_send',
      true
    );

    GENERATE_MSG(
      30000,
      "dio_4"
    );

  } else if (msg.equals("on_done_4")) {
    break;
  }
}

log.log(
  "Local Repair attack OFF from "
  + attacker.getID()
  + "!\n"
);

sim.getEventCentral().logEvent(
  "stop-attack",
  "localrepair:" + attacker.getID()
);

setBool(
  attacker,
  'network_attacks_rpl_dio_send',
  true
);

setBool(
  attacker,
  'network_attacks_rpl_dio_reset',
  true
);


/* Cycle 4: OFF */

GENERATE_MSG(900000, "off_done_4");

while (true) {
  YIELD();

  if (msg.equals("off_done_4")) {
    break;
  }
}

log.log(
  "Local Repair attack ON from "
  + attacker.getID()
  + "!\n"
);


/* Cycle 5: ON */

GENERATE_MSG(900000, "on_done_5");
GENERATE_MSG(30000, "dio_5");

while (true) {
  YIELD();

  if (msg.equals("dio_5")) {

    setBool(
      attacker,
      'network_attacks_local_repair_dio_send',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_reset',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_send',
      true
    );

    GENERATE_MSG(
      30000,
      "dio_5"
    );

  } else if (msg.equals("on_done_5")) {
    break;
  }
}

log.log(
  "Local Repair attack OFF from "
  + attacker.getID()
  + "!\n"
);

sim.getEventCentral().logEvent(
  "stop-attack",
  "localrepair:" + attacker.getID()
);

setBool(
  attacker,
  'network_attacks_rpl_dio_send',
  true
);

setBool(
  attacker,
  'network_attacks_rpl_dio_reset',
  true
);


/* Cycle 5: OFF */

GENERATE_MSG(900000, "off_done_5");

while (true) {
  YIELD();

  if (msg.equals("off_done_5")) {
    break;
  }
}

log.log(
  "Local Repair attack ON from "
  + attacker.getID()
  + "!\n"
);


/* Cycle 6: ON */

GENERATE_MSG(900000, "on_done_6");
GENERATE_MSG(30000, "dio_6");

while (true) {
  YIELD();

  if (msg.equals("dio_6")) {

    setBool(
      attacker,
      'network_attacks_local_repair_dio_send',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_reset',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_send',
      true
    );

    GENERATE_MSG(
      30000,
      "dio_6"
    );

  } else if (msg.equals("on_done_6")) {
    break;
  }
}

log.log(
  "Local Repair attack OFF from "
  + attacker.getID()
  + "!\n"
);

sim.getEventCentral().logEvent(
  "stop-attack",
  "localrepair:" + attacker.getID()
);

setBool(
  attacker,
  'network_attacks_rpl_dio_send',
  true
);

setBool(
  attacker,
  'network_attacks_rpl_dio_reset',
  true
);


/* Cycle 6: OFF */

GENERATE_MSG(900000, "off_done_6");

while (true) {
  YIELD();

  if (msg.equals("off_done_6")) {
    break;
  }
}

log.log(
  "Local Repair attack ON from "
  + attacker.getID()
  + "!\n"
);


/* Cycle 7: ON */

GENERATE_MSG(900000, "on_done_7");
GENERATE_MSG(30000, "dio_7");

while (true) {
  YIELD();

  if (msg.equals("dio_7")) {

    setBool(
      attacker,
      'network_attacks_local_repair_dio_send',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_reset',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_send',
      true
    );

    GENERATE_MSG(
      30000,
      "dio_7"
    );

  } else if (msg.equals("on_done_7")) {
    break;
  }
}

log.log(
  "Local Repair attack OFF from "
  + attacker.getID()
  + "!\n"
);

sim.getEventCentral().logEvent(
  "stop-attack",
  "localrepair:" + attacker.getID()
);

setBool(
  attacker,
  'network_attacks_rpl_dio_send',
  true
);

setBool(
  attacker,
  'network_attacks_rpl_dio_reset',
  true
);


/* Cycle 7: OFF */

GENERATE_MSG(900000, "off_done_7");

while (true) {
  YIELD();

  if (msg.equals("off_done_7")) {
    break;
  }
}

log.log(
  "Local Repair attack ON from "
  + attacker.getID()
  + "!\n"
);


/* Cycle 8: ON */

GENERATE_MSG(900000, "on_done_8");
GENERATE_MSG(30000, "dio_8");

while (true) {
  YIELD();

  if (msg.equals("dio_8")) {

    setBool(
      attacker,
      'network_attacks_local_repair_dio_send',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_reset',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_send',
      true
    );

    GENERATE_MSG(
      30000,
      "dio_8"
    );

  } else if (msg.equals("on_done_8")) {
    break;
  }
}

log.log(
  "Local Repair attack OFF from "
  + attacker.getID()
  + "!\n"
);

sim.getEventCentral().logEvent(
  "stop-attack",
  "localrepair:" + attacker.getID()
);

setBool(
  attacker,
  'network_attacks_rpl_dio_send',
  true
);

setBool(
  attacker,
  'network_attacks_rpl_dio_reset',
  true
);


/* Cycle 8: OFF */

GENERATE_MSG(900000, "off_done_8");

while (true) {
  YIELD();

  if (msg.equals("off_done_8")) {
    break;
  }
}

log.log(
  "Local Repair attack ON from "
  + attacker.getID()
  + "!\n"
);


/* Cycle 9: ON */

GENERATE_MSG(900000, "on_done_9");
GENERATE_MSG(30000, "dio_9");

while (true) {
  YIELD();

  if (msg.equals("dio_9")) {

    setBool(
      attacker,
      'network_attacks_local_repair_dio_send',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_reset',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_send',
      true
    );

    GENERATE_MSG(
      30000,
      "dio_9"
    );

  } else if (msg.equals("on_done_9")) {
    break;
  }
}

log.log(
  "Local Repair attack OFF from "
  + attacker.getID()
  + "!\n"
);

sim.getEventCentral().logEvent(
  "stop-attack",
  "localrepair:" + attacker.getID()
);

setBool(
  attacker,
  'network_attacks_rpl_dio_send',
  true
);

setBool(
  attacker,
  'network_attacks_rpl_dio_reset',
  true
);


/* Cycle 9: OFF */

GENERATE_MSG(900000, "off_done_9");

while (true) {
  YIELD();

  if (msg.equals("off_done_9")) {
    break;
  }
}

log.log(
  "Local Repair attack ON from "
  + attacker.getID()
  + "!\n"
);


/* Cycle 10: ON */

GENERATE_MSG(900000, "on_done_10");
GENERATE_MSG(30000, "dio_10");

while (true) {
  YIELD();

  if (msg.equals("dio_10")) {

    setBool(
      attacker,
      'network_attacks_local_repair_dio_send',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_reset',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_send',
      true
    );

    GENERATE_MSG(
      30000,
      "dio_10"
    );

  } else if (msg.equals("on_done_10")) {
    break;
  }
}

log.log(
  "Local Repair attack OFF from "
  + attacker.getID()
  + "!\n"
);

sim.getEventCentral().logEvent(
  "stop-attack",
  "localrepair:" + attacker.getID()
);

setBool(
  attacker,
  'network_attacks_rpl_dio_send',
  true
);

setBool(
  attacker,
  'network_attacks_rpl_dio_reset',
  true
);


/* Cycle 10: OFF */

GENERATE_MSG(900000, "off_done_10");

while (true) {
  YIELD();

  if (msg.equals("off_done_10")) {
    break;
  }
}

log.log(
  "Local Repair attack ON from "
  + attacker.getID()
  + "!\n"
);


/* Cycle 11: ON */

GENERATE_MSG(900000, "on_done_11");
GENERATE_MSG(30000, "dio_11");

while (true) {
  YIELD();

  if (msg.equals("dio_11")) {

    setBool(
      attacker,
      'network_attacks_local_repair_dio_send',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_reset',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_send',
      true
    );

    GENERATE_MSG(
      30000,
      "dio_11"
    );

  } else if (msg.equals("on_done_11")) {
    break;
  }
}

log.log(
  "Local Repair attack OFF from "
  + attacker.getID()
  + "!\n"
);

sim.getEventCentral().logEvent(
  "stop-attack",
  "localrepair:" + attacker.getID()
);

setBool(
  attacker,
  'network_attacks_rpl_dio_send',
  true
);

setBool(
  attacker,
  'network_attacks_rpl_dio_reset',
  true
);


/* Cycle 11: OFF */

GENERATE_MSG(900000, "off_done_11");

while (true) {
  YIELD();

  if (msg.equals("off_done_11")) {
    break;
  }
}

log.log(
  "Local Repair attack ON from "
  + attacker.getID()
  + "!\n"
);


/* Cycle 12: ON */

GENERATE_MSG(900000, "on_done_12");
GENERATE_MSG(30000, "dio_12");

while (true) {
  YIELD();

  if (msg.equals("dio_12")) {

    setBool(
      attacker,
      'network_attacks_local_repair_dio_send',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_reset',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_send',
      true
    );

    GENERATE_MSG(
      30000,
      "dio_12"
    );

  } else if (msg.equals("on_done_12")) {
    break;
  }
}

log.log(
  "Local Repair attack OFF from "
  + attacker.getID()
  + "!\n"
);

sim.getEventCentral().logEvent(
  "stop-attack",
  "localrepair:" + attacker.getID()
);

setBool(
  attacker,
  'network_attacks_rpl_dio_send',
  true
);

setBool(
  attacker,
  'network_attacks_rpl_dio_reset',
  true
);


/* Cycle 12: OFF */

GENERATE_MSG(900000, "off_done_12");

while (true) {
  YIELD();

  if (msg.equals("off_done_12")) {
    break;
  }
}

log.log(
  "Local Repair attack ON from "
  + attacker.getID()
  + "!\n"
);


/* Cycle 13: ON */

GENERATE_MSG(900000, "on_done_13");
GENERATE_MSG(30000, "dio_13");

while (true) {
  YIELD();

  if (msg.equals("dio_13")) {

    setBool(
      attacker,
      'network_attacks_local_repair_dio_send',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_reset',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_send',
      true
    );

    GENERATE_MSG(
      30000,
      "dio_13"
    );

  } else if (msg.equals("on_done_13")) {
    break;
  }
}

log.log(
  "Local Repair attack OFF from "
  + attacker.getID()
  + "!\n"
);

sim.getEventCentral().logEvent(
  "stop-attack",
  "localrepair:" + attacker.getID()
);

setBool(
  attacker,
  'network_attacks_rpl_dio_send',
  true
);

setBool(
  attacker,
  'network_attacks_rpl_dio_reset',
  true
);


/* Cycle 13: OFF */

GENERATE_MSG(900000, "off_done_13");

while (true) {
  YIELD();

  if (msg.equals("off_done_13")) {
    break;
  }
}

log.log(
  "Local Repair attack ON from "
  + attacker.getID()
  + "!\n"
);


/* Cycle 14: ON */

GENERATE_MSG(900000, "on_done_14");
GENERATE_MSG(30000, "dio_14");

while (true) {
  YIELD();

  if (msg.equals("dio_14")) {

    setBool(
      attacker,
      'network_attacks_local_repair_dio_send',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_reset',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_send',
      true
    );

    GENERATE_MSG(
      30000,
      "dio_14"
    );

  } else if (msg.equals("on_done_14")) {
    break;
  }
}

log.log(
  "Local Repair attack OFF from "
  + attacker.getID()
  + "!\n"
);

sim.getEventCentral().logEvent(
  "stop-attack",
  "localrepair:" + attacker.getID()
);

setBool(
  attacker,
  'network_attacks_rpl_dio_send',
  true
);

setBool(
  attacker,
  'network_attacks_rpl_dio_reset',
  true
);


/* Cycle 14: OFF */

GENERATE_MSG(900000, "off_done_14");

while (true) {
  YIELD();

  if (msg.equals("off_done_14")) {
    break;
  }
}

log.log(
  "Local Repair attack ON from "
  + attacker.getID()
  + "!\n"
);


/* Cycle 15: ON */

GENERATE_MSG(900000, "on_done_15");
GENERATE_MSG(30000, "dio_15");

while (true) {
  YIELD();

  if (msg.equals("dio_15")) {

    setBool(
      attacker,
      'network_attacks_local_repair_dio_send',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_reset',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_send',
      true
    );

    GENERATE_MSG(
      30000,
      "dio_15"
    );

  } else if (msg.equals("on_done_15")) {
    break;
  }
}

log.log(
  "Local Repair attack OFF from "
  + attacker.getID()
  + "!\n"
);

sim.getEventCentral().logEvent(
  "stop-attack",
  "localrepair:" + attacker.getID()
);

setBool(
  attacker,
  'network_attacks_rpl_dio_send',
  true
);

setBool(
  attacker,
  'network_attacks_rpl_dio_reset',
  true
);


/* Cycle 15: OFF */

GENERATE_MSG(900000, "off_done_15");

while (true) {
  YIELD();

  if (msg.equals("off_done_15")) {
    break;
  }
}

log.log(
  "Local Repair attack ON from "
  + attacker.getID()
  + "!\n"
);


/* Cycle 16: ON */

GENERATE_MSG(900000, "on_done_16");
GENERATE_MSG(30000, "dio_16");

while (true) {
  YIELD();

  if (msg.equals("dio_16")) {

    setBool(
      attacker,
      'network_attacks_local_repair_dio_send',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_reset',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_send',
      true
    );

    GENERATE_MSG(
      30000,
      "dio_16"
    );

  } else if (msg.equals("on_done_16")) {
    break;
  }
}

log.log(
  "Local Repair attack OFF from "
  + attacker.getID()
  + "!\n"
);

sim.getEventCentral().logEvent(
  "stop-attack",
  "localrepair:" + attacker.getID()
);

setBool(
  attacker,
  'network_attacks_rpl_dio_send',
  true
);

setBool(
  attacker,
  'network_attacks_rpl_dio_reset',
  true
);


/* Cycle 16: OFF */

GENERATE_MSG(900000, "off_done_16");

while (true) {
  YIELD();

  if (msg.equals("off_done_16")) {
    break;
  }
}

log.log(
  "Local Repair attack ON from "
  + attacker.getID()
  + "!\n"
);


/* Cycle 17: ON */

GENERATE_MSG(900000, "on_done_17");
GENERATE_MSG(30000, "dio_17");

while (true) {
  YIELD();

  if (msg.equals("dio_17")) {

    setBool(
      attacker,
      'network_attacks_local_repair_dio_send',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_reset',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_send',
      true
    );

    GENERATE_MSG(
      30000,
      "dio_17"
    );

  } else if (msg.equals("on_done_17")) {
    break;
  }
}

log.log(
  "Local Repair attack OFF from "
  + attacker.getID()
  + "!\n"
);

sim.getEventCentral().logEvent(
  "stop-attack",
  "localrepair:" + attacker.getID()
);

setBool(
  attacker,
  'network_attacks_rpl_dio_send',
  true
);

setBool(
  attacker,
  'network_attacks_rpl_dio_reset',
  true
);


/* Cycle 17: OFF */

GENERATE_MSG(900000, "off_done_17");

while (true) {
  YIELD();

  if (msg.equals("off_done_17")) {
    break;
  }
}

log.log(
  "Local Repair attack ON from "
  + attacker.getID()
  + "!\n"
);


/* Cycle 18: ON */

GENERATE_MSG(900000, "on_done_18");
GENERATE_MSG(30000, "dio_18");

while (true) {
  YIELD();

  if (msg.equals("dio_18")) {

    setBool(
      attacker,
      'network_attacks_local_repair_dio_send',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_reset',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_send',
      true
    );

    GENERATE_MSG(
      30000,
      "dio_18"
    );

  } else if (msg.equals("on_done_18")) {
    break;
  }
}

log.log(
  "Local Repair attack OFF from "
  + attacker.getID()
  + "!\n"
);

sim.getEventCentral().logEvent(
  "stop-attack",
  "localrepair:" + attacker.getID()
);

setBool(
  attacker,
  'network_attacks_rpl_dio_send',
  true
);

setBool(
  attacker,
  'network_attacks_rpl_dio_reset',
  true
);


/* Cycle 18: OFF */

GENERATE_MSG(900000, "off_done_18");

while (true) {
  YIELD();

  if (msg.equals("off_done_18")) {
    break;
  }
}

log.log(
  "Local Repair attack ON from "
  + attacker.getID()
  + "!\n"
);


/* Cycle 19: ON */

GENERATE_MSG(900000, "on_done_19");
GENERATE_MSG(30000, "dio_19");

while (true) {
  YIELD();

  if (msg.equals("dio_19")) {

    setBool(
      attacker,
      'network_attacks_local_repair_dio_send',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_reset',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_send',
      true
    );

    GENERATE_MSG(
      30000,
      "dio_19"
    );

  } else if (msg.equals("on_done_19")) {
    break;
  }
}

log.log(
  "Local Repair attack OFF from "
  + attacker.getID()
  + "!\n"
);

sim.getEventCentral().logEvent(
  "stop-attack",
  "localrepair:" + attacker.getID()
);

setBool(
  attacker,
  'network_attacks_rpl_dio_send',
  true
);

setBool(
  attacker,
  'network_attacks_rpl_dio_reset',
  true
);


/* Cycle 19: OFF */

GENERATE_MSG(900000, "off_done_19");

while (true) {
  YIELD();

  if (msg.equals("off_done_19")) {
    break;
  }
}

log.log(
  "Local Repair attack ON from "
  + attacker.getID()
  + "!\n"
);


/* Cycle 20: ON */

GENERATE_MSG(900000, "on_done_20");
GENERATE_MSG(30000, "dio_20");

while (true) {
  YIELD();

  if (msg.equals("dio_20")) {

    setBool(
      attacker,
      'network_attacks_local_repair_dio_send',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_reset',
      true
    );

    setBool(
      attacker,
      'network_attacks_rpl_dio_send',
      true
    );

    GENERATE_MSG(
      30000,
      "dio_20"
    );

  } else if (msg.equals("on_done_20")) {
    break;
  }
}

log.log(
  "Local Repair attack OFF from "
  + attacker.getID()
  + "!\n"
);

sim.getEventCentral().logEvent(
  "stop-attack",
  "localrepair:" + attacker.getID()
);

setBool(
  attacker,
  'network_attacks_rpl_dio_send',
  true
);

setBool(
  attacker,
  'network_attacks_rpl_dio_reset',
  true
);


/* Cycle 20: OFF */

GENERATE_MSG(900000, "off_done_20");

while (true) {
  YIELD();

  if (msg.equals("off_done_20")) {
    break;
  }
}

log.log(
  "Local Repair attack ON from "
  + attacker.getID()
  + "!\n"
);


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