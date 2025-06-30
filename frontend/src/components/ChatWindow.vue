<template>
  <v-card color="surface" class="pa-4 d-flex flex-column" style="height: 600px;" elevation="4">
    <h3 class="text-h6 font-weight-bold mb-2">Chat with RobotDoc</h3>

    <!-- Chat message history -->
    <div
      ref="chatContainer"
      class="flex-grow-1 overflow-y-auto mb-2 pr-1 d-flex flex-column"
      style="min-height: 0;"
    >
      <div
        v-for="(msg, i) in messages"
        :key="i"
        class="mb-2 px-3 py-2 rounded"
        :class="msg.from === 'You' ? 'align-self-end bg-primary text-white' : 'align-self-start bg-grey-darken-3'"
        style="max-width: 75%; white-space: pre-line;"
      >
        <strong v-if="msg.from !== 'You'" class="mr-1">{{ msg.from }}:</strong>{{ msg.text }}
      </div>
    </div>

    <!-- Checkboxes and input field -->
    <div class="d-flex flex-column mt-2">
      <div class="d-flex mb-2">
        <v-checkbox
          v-model="updateSymptoms"
          label="Update Symptoms"
          density="compact"
          hide-details
        />
        <v-checkbox
          v-model="useKG"
          label="Use Knowledge Graph"
          density="compact"
          hide-details
        />
      </div>

      <div class="d-flex align-center">
        <v-text-field
          v-model="newMessage"
          label="Type a message..."
          class="flex-grow-1"
          style="min-height: 56px"
          @keyup.enter="sendMessage"
          clearable
        />
        <v-btn color="primary" class="ml-2" @click="sendMessage">Send</v-btn>
      </div>
    </div>
  </v-card>
</template>

<script setup>
import { ref, nextTick, watch } from 'vue'

const props = defineProps({
  patientId: Number
})

const messages = ref([
  { from: 'Doctor', text: 'How are you feeling today?' },
  { from: 'Patient', text: 'Tired but okay.' }
])

const newMessage = ref('')
const updateSymptoms = ref(true)
const useKG = ref(true)
const chatContainer = ref(null)

const API_BASE_URL = import.meta.env.VITE_BACKEND_URL

function scrollToBottom() {
  nextTick(() => {
    if (chatContainer.value) {
      chatContainer.value.scrollTop = chatContainer.value.scrollHeight
    }
  })
}

watch(messages, scrollToBottom, { deep: true })

async function sendMessage() {
  if (!newMessage.value.trim()) return

  const input = newMessage.value
  messages.value.push({ from: 'You', text: input })
  newMessage.value = ''

  try {
    const response = await fetch(`${API_BASE_URL}/api/respond/${props.patientId}`, {
      method: 'POST',
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        message: input,
        updateSymptoms: updateSymptoms.value,
        useKG: useKG.value
      })
    })

    const data = await response.json()
    if (data.reply) {
      if (data.type === 'symptoms') {
        messages.value.push({ from: 'System', text: 'Detected new symptoms: ' + data.reply.join(', ') })
      } else {
        messages.value.push({ from: 'System', text: data.reply })
      }
    } else {
      messages.value.push({ from: 'System', text: 'No reply from model.' })
    }
  } catch (error) {
    messages.value.push({ from: 'Error', text: 'Failed to send message: ' + error.message })
  }
}
</script>
