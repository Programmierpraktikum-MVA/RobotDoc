<template>
  <v-card color="surface" class="pa-4 d-flex flex-column position-relative" style="height: 600px;" elevation="4">
    <!-- Floating Options Button -->
    <v-menu location="bottom end">
      <template #activator="{ props }">
        <v-btn icon variant="text" v-bind="props" class="position-absolute" style="top: 12px; right: 12px;" aria-label="Options">
          <v-icon>mdi-dots-vertical</v-icon>
        </v-btn>
      </template>
      <v-card min-width="220" class="pa-2">
        <v-list density="compact">
          <v-list-item>
            <v-checkbox v-model="useKG" label="Use Knowledge Graph" density="comfortable" hide-details />
          </v-list-item>
          <v-list-item>
            <v-checkbox v-model="updateSymptoms" label="Update Symptoms" density="comfortable" hide-details />
          </v-list-item>
        </v-list>
      </v-card>
    </v-menu>

    <!-- Header -->
    <div class="d-flex align-center mb-4">
      <v-avatar size="36" class="mr-3"><v-icon color="primary">mdi-robot</v-icon></v-avatar>
      <div>
        <div class="text-subtitle-1 font-weight-bold">RobotDoc</div>
        <div class="text-caption d-flex align-center text-green">
          <v-icon size="10" class="mr-1">mdi-circle</v-icon>Online
        </div>
      </div>
    </div>

    <!-- Chat Messages -->
    <div ref="chatContainer" class="flex-grow-1 overflow-y-auto mb-2 pr-1 d-flex flex-column" style="min-height: 0;">
      <div v-for="(msg, i) in displayedMessages" :key="i"
           class="mb-2 px-3 py-2 rounded"
           :class="msg.from === 'You' ? 'align-self-end bg-primary text-white' : 'align-self-start bg-grey-darken-3 text-white'"
           style="max-width: 75%; white-space: pre-line;">
        <template v-if="msg.text === '...'">
          <div class="typing-indicator">
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
          </div>
        </template>
        <template v-else>
          <div v-if="msg.imageUrl" class="mb-2">
            <v-img :src="msg.imageUrl" max-width="200" max-height="200" class="mb-2 rounded" />
          </div>
          <div v-if="msg.text">{{ msg.text }}</div>
          <div class="text-caption mt-1" style="font-size: 0.7rem; opacity: 0.6;">
            {{ formatTime(msg.time) }}
          </div>
        </template>
      </div>
    </div>

    <!-- Input & Actions -->
    <div class="d-flex align-center">
      <v-text-field
        v-model="newMessage"
        label="Type a message..."
        class="flex-grow-1"
        variant="outlined"
        rounded="xl"
        density="comfortable"
        hide-details
        @keyup.enter="sendMessage"
        clearable
      />
      <v-btn icon color="primary" class="ml-2" elevation="2" @click="sendMessage">
        <v-icon>mdi-send</v-icon>
      </v-btn>
      <v-btn icon class="ml-2" @click="triggerFileInput">
        <v-icon>mdi-camera</v-icon>
      </v-btn>
      <input type="file" ref="fileInput" accept="image/*" style="display: none" @change="onImageSelected" />
    </div>

    <!-- Image Preview -->
    <div v-if="selectedImage" class="mt-2 d-flex align-center">
      <v-img :src="imagePreviewUrl" max-height="120" max-width="120" class="mr-3 rounded" />
      <v-btn variant="text" icon @click="cancelImage"><v-icon>mdi-close</v-icon></v-btn>
    </div>
  </v-card>
</template>

<script setup>
import { ref, watch, computed, nextTick } from 'vue'

const props = defineProps({ patientId: Number })

const messages = ref([])
const newMessage = ref('')
const updateSymptoms = ref(true)
const useKG = ref(true)
const chatContainer = ref(null)
const systemTyping = ref(false)

const selectedImage = ref(null)
const imagePreviewUrl = ref(null)
const fileInput = ref(null)

const API_BASE_URL = import.meta.env.VITE_BACKEND_URL

const youAreTyping = computed(() => newMessage.value.trim().length > 0)

const displayedMessages = computed(() => {
  const result = [...messages.value]
  if (youAreTyping.value) result.push({ from: 'You', text: '...' })
  if (systemTyping.value) result.push({ from: 'System', text: '...' })
  return result
})

watch(messages, () => nextTick(() => {
  if (chatContainer.value) {
    chatContainer.value.scrollTop = chatContainer.value.scrollHeight
  }
}), { deep: true })

watch(() => props.patientId, (newId) => {
  if (newId) fetchChatHistory()
}, { immediate: true })

async function fetchChatHistory() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/chat_history/${props.patientId}`, {
      credentials: 'include'
    })
    const history = await res.json()
    messages.value = history.map(entry => ({
      from: entry.role === 'user' ? 'You' : 'System',
      text: entry.content,
      time: entry.timestamp ? new Date(entry.timestamp) : new Date()
    }))
  } catch (err) {
    console.error("Failed to fetch chat history:", err)
  }
}

function formatTime(date) {
  if (!(date instanceof Date) || isNaN(date)) return ''
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function triggerFileInput() {
  fileInput.value?.click()
}

function onImageSelected(event) {
  const file = event.target.files?.[0]
  if (!file) return
  selectedImage.value = file
  imagePreviewUrl.value = URL.createObjectURL(file)
  event.target.value = null
}

function cancelImage() {
  selectedImage.value = null
  if (imagePreviewUrl.value) URL.revokeObjectURL(imagePreviewUrl.value)
  imagePreviewUrl.value = null
}

async function sendMessage() {
  if (!newMessage.value.trim() && !selectedImage.value) return

  const input = newMessage.value.trim()
  const timestamp = new Date()

  if (selectedImage.value) {
    const file = selectedImage.value
    const imageUrl = URL.createObjectURL(file)

    messages.value.push({ from: 'You', text: input, imageUrl, time: timestamp })

    const formData = new FormData()
    formData.append('image', file)
    formData.append('imgcontext', input)

    newMessage.value = ''
    selectedImage.value = null
    imagePreviewUrl.value = null
    systemTyping.value = true

    try {
      const res = await fetch(`${API_BASE_URL}/api/llava/uploadImageForPatient/${props.patientId}`, {
        method: 'POST',
        credentials: 'include',
        body: formData
      })

      const data = await res.json()
      systemTyping.value = false

      messages.value.push({
        from: 'System',
        text: data.reply || 'No reply from model.',
        time: new Date()
      })
    } catch (err) {
      systemTyping.value = false
      messages.value.push({ from: 'Error', text: 'Upload failed: ' + err.message, time: new Date() })
    }

  } else {
    messages.value.push({ from: 'You', text: input, time: timestamp })
    newMessage.value = ''
    systemTyping.value = true

    try {
      const response = await fetch(`${API_BASE_URL}/api/respond/${props.patientId}`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: input, updateSymptoms: updateSymptoms.value, useKG: useKG.value })
      })

      const data = await response.json()
      systemTyping.value = false

      const replyText = data.type === 'symptoms'
        ? 'Detected new symptoms: ' + data.reply.join(', ')
        : data.reply || 'No reply from model.'

      messages.value.push({ from: 'System', text: replyText, time: new Date() })

    } catch (error) {
      systemTyping.value = false
      messages.value.push({ from: 'Error', text: 'Failed to send message: ' + error.message, time: new Date() })
    }
  }
}
</script>

<style scoped>
.typing-indicator {
  display: flex;
  align-items: center;
  gap: 6px;
  height: 20px;
}
.typing-dot {
  width: 8px;
  height: 8px;
  background-color: white;
  border-radius: 50%;
  animation: blink 1.4s infinite both;
  opacity: 0.3;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink {
  0%, 80%, 100% { opacity: 0.2; }
  40% { opacity: 1; }
}
.text-green { color: #4CAF50; }
</style>
