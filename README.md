# Installing Contiki-NG and Cooja
Based on the Contiki-NG documentation available: https://docs.contiki-ng.org/en/master/doc/getting-started/Toolchain-installation-on-Linux.html

The below instructions are intended for the university server where parts of the installation is already done. To install elsewhere, use the referenced documentation above and simply jump to the final step of the instructions below.

1. Add yourself to the wireshark group:
```console
$ sudo usermod -a -G wireshark <user>
```
2. Install ARM compiler in your home directory
```console
$ cd ~
$ wget https://developer.arm.com/-/media/Files/downloads/gnu-rm/10.3-2021.10/gcc-arm-none-eabi-10.3-2021.10-x86_64-linux.tar.bz2
$ tar -xjf gcc-arm-none-eabi-10.3-2021.10-x86_64-linux.tar.bz2
```

3. Configure .bahsrc
Append the following to your .bashrc (replace <user> with your user):\
export PATH="$PATH:/home/<user>/gcc-arm-none-eabi-10.3-2021.10/bin"\
export JAVA_HOME="/usr/lib/jvm/default-java"
```console
$ cd ~
$ nano .bashrc
(append the two lines)
```

4. Clone this repo along with the submodules (cooja and contiki-ng)
```console
$ git clone --recurse-submodules -j8 https://github.com/uu-core/ids-WPLR
```


# RPL Attack Simulation using Cooja (Contiki-NG)

This repository builds on the multi-trace repository to provide a framework for simulating network-layer attacks on the RPL protocol using the Cooja simulator within the Contiki-NG operating system. The project supports automated scenario generation, modular attack implementations, and structured application-layer simulations.

## Repository Structure

├── applications/\
├── services/\
├── node\_generation/\
├── tools/\
└── csv\_generation/

### `applications/`

This folder contains all Cooja scenario (`.csc`) files and application-level code for data transmission.

- **Cooja Scenario Files**: Define the network topology, node positions, and simulation parameters (e.g., attack start/stop times, number of nodes).
- **UDP Applications**:
  - `udp-client`: Sends application-layer data from regular nodes.
  - `udp-server`: Receives data, typically deployed on sink/root nodes.
- These files also control runtime behavior by setting flags used to trigger specific attacks (defined in `services/`).

### `services/`

This directory includes the C-based implementations of various RPL attacks.

- Each attack is written as a Contiki process and compiled with the rest of the Contiki-NG system.
- The behavior of each attack is controlled by a runtime flag (a boolean value).
- During simulation, each node’s process continuously checks whether its corresponding attack flag is active. If so, it executes the attack logic.
- This approach allows dynamic enabling/disabling of attacks from within `.csc` scenario files without recompilation.

### `node_generation/`

This folder automates the generation of scenarios.

- **Python Scripts**:
  - Generate node position JSON files based on desired network topologies.
  - Insert these positions into template `.csc` files.
- Automates creation of simulations with different node counts, attack types, and variations.

### `tools/`

Contains the cooja simulator and the contiki-ng operating system, files in here should generally not be modified.

### `csv_generation/`

This folder deals with creating the csv files from the scenario outputs

- **Python Scripts**:
  - Process the log files in an output folder into a csv file that can be used as features for the ML models.
  - Automate the process of going through all output folders.
---

## Example Run

1. Navigate to the `node_generation/` folder and run the files `gen_nodes.py` and `insert_nodes.py`. `gen_nodes.py` takes a list of integers as arguemnt, that is the number of nodes to be generated. `insert_nodes.py` takes a list of strings, that is the attacks the generated nodes should be insterted into.
```console
$ cd node_generation
$ python3 gen_nodes.py 5,10,15,20
$ python3 insert_nodes.py worst_parent,local_repair,blackhole,dis_flooding,failing_node
```
This will have created the `applications/example-attacks/scenarios/` folder which contains all of the generated .csc files

2. To run the experiments, use the `run-experiments.sh` script. By default it will run all generated .csc files below `applications/example-attacks/scenarios/`, you can also specify if you want to run a specific attack/size/variations by using the arguments --attack --size --variation .\
For example, the command:
```console
$ ./run-experiments.sh --attack blackhole
```
will run all the blackhole scenarios.\
The command:
```console
$ ./run-experiments.sh --size 10 --variation base
```
will run the scenarios of size 10 and of the base variation. \
The command:
```console
$ ./run-experiments.sh
```
will run all scenarios. \
If you want to run scenarios in the background; use `nohup` with an appropriate timeout:
```console
$ nohup timeout 4h ./run-experiments.sh --attack blackhole > output.log 2>&1 &
```
will run all the blackhole scenarios in the background and write any outputs to `output.log`, and will kill the process if it is still running after 4 hours.\
\
Note that the arguments must match the folder names under the `applications/example-attacks/scenarios/` folder.

3. The outputs of the experiments are placed under `applications/example-attacks/scenarios_output/` with the same folder structure as `applications/example-attacks/scenarios/`

4. Finally, to generate the .csv files used as the machine learning model features, run the `csv_generation/gen_csv.py` script, it takes a list of attack types as argument (must match the folder names under `applications/example-attacks/scenarios_output/`). This will place the csv files in the output folders located in `applications/example-attacks/scenarios_output/`.
```console
$ python3 csv_generation/gen_csv.py worst_parent,local_repair
```
---
## Adding a New Attack

1. Navigate to the `services/network-attacks` folder.
2. In `network-attacks.c`, add a runtime check for your attack's control flag:

   ```c
   bool attack_enabled_flag = false;
   static void
    check_config(void *ptr)
    {
    if (attack_enabled_flag) {
        // Attack behavior function call here
    }
    }
   ```
   Some attacks require the contiki-ng operating system to be modified beyond simply modifying the handling of outgoing/incoming traffic, like for the Worst Parent attack. The first step is to identify the part of OS to be modified, which requires studying the OS implementation in `tools/contiki-ng/os`. The implementation of the Worst Parent attack can be used as a reference for how to proceed once the relevant part of the OS is found.
3. Update the relevant `.csc` files in `applications/` to include and toggle your attack.

### Example Implementation: Failing Node
The purpose of the failing node scenario is to generate data that is out of the ordinary but not due to an attack. Instead, a node will "fail", by "fail" we mean going offline, and from the perspective of the other nodes in the network the failing node disappears. This data can then be used to evaluate if a model is actually detecting attacks, or if it only detects when the data is not normal.

The way we simulate this behavior in the simulator is by shutting off the radio of the node chosen for failiure, which will stop all outgoing communication from the node and make it invisible from other nodes.

1. Find out where in `tools/contiki-ng/os` the relevant code is for manipulating the radio of a node: On API for toggling the radio can be found in `tools/contiki-ng/os/dev/radio.h`.
2. Add the logic of toggling the radio and the runtime flag check in `network-attacks.c`:
```c
...

#include "dev/radio.h"
...

bool network_attacks_toggle_radio = false;
static bool radio_is_on = true;
...

static void
toggle_radio(void) { //
  network_attacks_toggle_radio = false;
  if(!radio_is_on) {
    NETSTACK_RADIO.on();
    radio_is_on = true;
  }
  else {
    NETSTACK_RADIO.off();
    radio_is_on = false;
  }
}
...

static void
  check_config(void *ptr) {
    ...
    if(network_attacks_toggle_radio) {
      toggle_radio();
    }
    ...
  }
```
3. In `applications/example-attacks/simulations`, create the `failing_node-base.csc` file, this is easiest done by copying one of the existing `.csc` as most of the configuation will be the same across all `.csc` files. You probably only need to modify the `<script>` part of the file, note that the function selectAttacker() is overwritten by `node_generation/insert_nodes.py`, so remember to modify `insert_nodes.py` if you want to make changes to selectAttacker().

To toggle activate your attack, you must set the bool that you added in `network-attacks.c` during the previous step using setBool(node id, bool name, value).
```c
setBool(attacker, 'network_attacks_toggle_radio', true);
```
---
