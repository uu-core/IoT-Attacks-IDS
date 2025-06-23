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
└── node\_generation/

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

---

## Extending the Codebase

### Adding a New Attack

1. **Create a new `.c` file** in `services/` implementing your attack logic.
2. Add a runtime check for your attack's control flag:

   ```c
   if (attack_enabled_flag) {
       // Attack behavior here
   }
   ```
3. Update the relevant `.csc` files in `applications/` to include and toggle your attack.

### Adding a New Scenario

1. Add your `.csc` template to `applications/`.
2. Modify or extend the Python scripts in `node_generation/` to:

   * Insert positions
   * Configure node types

---

## Notes

* Attack toggling is handled entirely through simulation configuration (`.csc`) files without requiring recompilation.
* Node behaviors (client, server, attacker) are defined by application roles and services enabled at runtime.

---

## License

---
