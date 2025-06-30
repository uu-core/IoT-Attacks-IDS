# Installing Contiki-NG and Cooja
Based on the Contiki-NG documentation available: https://docs.contiki-ng.org/en/master/doc/getting-started/index.html

The below instructions are intended for the server where parts of the installation is already done. To install elsewhere, use the referenced documentation above.

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
$ python3 gen_nodes.py '[5, 10, 15, 20]'
$ python3 insert_nodes.py '["worst_parent", "local_repair", "blackhole", "dis_flooding"]'
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
Note that the arguments must match the folder names under the `applications/example-attacks/scenarios/` folder.

3. The outputs of the experiments are placed under `applications/example-attacks/scenarios_output/` with the same folder structure as `applications/example-attacks/scenarios/`

4. Finally, to generate the .csv files used as the machine learning model features, run the `csv_generation/gen_csv.py` script, it takes a list of attack types as argument (must match the folder names under `applications/example-attacks/scenarios_output/`). This will place the csv files in the output folders located in `applications/example-attacks/scenarios_output/`.
```console
$ python3 csv_generation/gen_csv.py '["worst_parent", "local repair"]'
```

### Adding a New Attack
1. Navigate to the `services/network-attacks` folder.
2. Add a runtime check for your attack's control flag:

   ```c
   bool attack_enabled_flag = false;
   static void
    check_config(void *ptr)
    {
    if (attack_enabled_flag) {
        // Attack behavior here
    }
    }
   ```
3. Update the relevant `.csc` files in `applications/` to include and toggle your attack.

---

## License

---
