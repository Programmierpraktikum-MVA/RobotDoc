<template>
  <v-card class="mt-4">
    <v-card-title>Chat</v-card-title>
    <v-card-text>
      <div style="max-height: 200px; overflow-y: auto">
        <div v-for="(msg, i) in messages" :key="i">
          <strong>{{ msg.from }}:</strong> {{ msg.text }}
        </div>
      </div>
      <v-text-field
        v-model="newMessage"
        label="Your message"
        @keyup.enter="sendMessage"
        clearable
      />
    </v-card-text>
    <v-card-actions>
      <v-btn color="primary" @click="sendMessage">Send</v-btn>
    </v-card-actions>
  </v-card>
</template>

<script setup>
import { ref } from 'vue'

const props = defineProps({
  patientId: Number
})

const messages = ref([
  { from: 'Doctor', text: 'How are you feeling today?' },
  { from: 'Patient', text: 'Tired but okay.' }
])

const newMessage = ref('')
const API_BASE_URL = import.meta.env.VITE_BACKEND_URL
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
        updateSymptoms: true,
        useKG: true
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
