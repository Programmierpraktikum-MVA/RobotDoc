<template>
  <v-app>
    <Navbar />
    <v-main>
      <v-container class="mt-6">
        <v-row>
          <!-- Patient Details Panel -->
          <v-col cols="12" md="6">
            <v-card color="surface" class="pa-6" elevation="4">
              <h2 class="text-h5 font-weight-bold mb-4">Patient Details</h2>
              <v-list dense>
                <v-list-item><v-list-item-title><strong>ID:</strong> {{ patient.id }}</v-list-item-title></v-list-item>
                <v-list-item><v-list-item-title><strong>Name:</strong> {{ patient.name }}</v-list-item-title></v-list-item>
                <v-list-item><v-list-item-title><strong>Age:</strong> {{ patient.age }}</v-list-item-title></v-list-item>
                <v-list-item><v-list-item-title><strong>Weight:</strong> {{ patient.weight }} kg</v-list-item-title></v-list-item>
                <v-list-item><v-list-item-title><strong>Sex:</strong> {{ patient.sex }}</v-list-item-title></v-list-item>
                <v-list-item>
                  <v-list-item-title>
                    <strong>Symptoms:</strong>
                    {{ Array.isArray(patient.symptoms) ? patient.symptoms.filter(Boolean).join(', ') : patient.symptoms }}
                  </v-list-item-title>
                </v-list-item>
              </v-list>
            </v-card>
          </v-col>

          <!-- Chat Panel -->
          <v-col cols="12" md="6">
            <v-card color="surface" class="pa-4 d-flex flex-column" style="height: 600px;" elevation="4">
              <h3 class="text-h6 font-weight-bold mb-2">Chat with RobotDoc</h3>
              <div
                ref="chatContainer"
                class="flex-grow-1 overflow-y-auto mb-2 pr-1 d-flex flex-column"
                style="min-height: 0;"
              >
                <div
                  v-for="(msg, i) in messages"
                  :key="i"
                  class="mb-2 px-3 py-2 rounded"
                  :class="msg.from === 'RoboDoc' ? 'align-self-start bg-grey-darken-3' : 'align-self-end bg-primary text-white'"
                  style="max-width: 75%; white-space: pre-line;"
                >
                  {{ msg.text }}
                </div>
              </div>
              <div class="d-flex align-center">
                <v-text-field
                  v-model="newMessage"
                  label="Type a message..."
                  class="mt-2 flex-grow-1"
                  style="min-height: 56px"
                  @keyup.enter="sendMessage"
                />
                <v-btn color="primary" class="ml-2 mt-2" @click="sendMessage">Send</v-btn>
              </div>
            </v-card>
          </v-col>
        </v-row>
      </v-container>
    </v-main>
  </v-app>
</template>

<script setup>
import { ref, onMounted, nextTick } from 'vue'
import { useRoute } from 'vue-router'
import axios from 'axios'
import Navbar from '@/components/Navbar.vue'

const route = useRoute()

const patient = ref({
  id: null,
  name: '',
  age: '',
  weight: '',
  sex: '',
  symptoms: []
})

const messages = ref([
  { from: 'RoboDoc', text: 'Hello, how are you feeling today?' },
  { from: 'Patient', text: 'A bit tired and dizzy.' }
])

const newMessage = ref('')
const chatContainer = ref(null)
const API_BASE_URL = import.meta.env.VITE_BACKEND_URL
function scrollToBottom() {
  nextTick(() => {
    if (chatContainer.value) {
      chatContainer.value.scrollTop = chatContainer.value.scrollHeight
    }
  })
}

function sendMessage() {
  if (newMessage.value.trim()) {
    messages.value.push({ from: 'You', text: newMessage.value })
    messages.value.push({ from: 'RoboDoc', text: 'Thanks for your update. We will monitor that.' })
    newMessage.value = ''
    scrollToBottom()
  }
}

onMounted(async () => {
  const id = Number(route.params.id)
  try {
    const response = await axios.get(`${API_BASE_URL}/api/patient/${id}`)
    patient.value = response.data
  } catch (error) {
    console.error('Failed to fetch patient:', error)
  }
  scrollToBottom()
})
</script>
