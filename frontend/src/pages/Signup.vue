<template>
  <v-container class="fill-height d-flex align-center justify-center">
    <v-card color="surface" class="pa-6" width="400" elevation="10">
      <div class="text-h4 font-weight-bold text-center mb-6">Sign Up</div>
      <v-card-text>
        <v-text-field
          v-model="username"
          label="Username"
          prepend-inner-icon="mdi-account"
        />
        <v-text-field
          v-model="password"
          label="Password"
          type="password"
          prepend-inner-icon="mdi-lock"
        />
        <div v-if="message" :class="messageType === 'error' ? 'text-error' : 'text-success'" class="text-caption mt-1 mb-3">
          {{ message }}
        </div>
      </v-card-text>
      <v-card-actions class="d-flex flex-column">
        <v-btn color="primary" class="mb-2" block @click="register">
          Create Account
        </v-btn>
        <v-btn variant="outlined" color="secondary" block @click="goToLogin">
          Already have an account?
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-container>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import axios from 'axios'

const username = ref('')
const password = ref('')
const message = ref('')
const messageType = ref('')
const router = useRouter()
const API_BASE_URL = import.meta.env.VITE_BACKEND_URL

async function register() {
  message.value = ''
  messageType.value = ''

  if (!username.value || !password.value) {
    message.value = 'Please enter both username and password.'
    messageType.value = 'error'
    return
  }

  try {
    await axios.post(`${API_BASE_URL}/api/register`, {
      username: username.value,
      password: password.value
    })
    message.value = 'Account created successfully!'
    messageType.value = 'success'
    setTimeout(() => router.push('/login'), 1500)
  } catch (err) {
    console.error(err)
    messageType.value = 'error'
    if (err.response?.data?.error) {
      message.value = `Error: ${err.response.data.error}`
    } else {
      message.value = 'Registration failed. See console for details.'
    }
  }
}

function goToLogin() {
  router.push('/login')
}
</script>
