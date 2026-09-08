#ifndef EXTENDED_MESSAGE_H_
#define EXTENDED_MESSAGE_H_

#include "app-message.h"

/**
 * Extended message structure for node-to-node communication
 * This preserves compatibility with app-message.h
 */
typedef struct {
  app_message_t base_msg;    // Original message structure
  uint8_t hop_count;         // Estimated hop count
  uint8_t source_id;         // Source node ID
  uint8_t target_id;         // Target node ID
  uint8_t reserved;          // For alignment
} extended_message_t;

#endif /* EXTENDED_MESSAGE_H_ */