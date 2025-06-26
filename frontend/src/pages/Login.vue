<template>
  <v-container class="fill-height d-flex align-center justify-center">
    <v-card color="surface" class="pa-6" width="400" elevation="10">
      <div class="text-h4 font-weight-bold text-center mb-6">RobotDoc</div>
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
        <div v-if="errorMessage" class="text-error text-caption mt-1 mb-3">
          {{ errorMessage }}
        </div>
        <v-row align="center" justify="space-between" class="mt-2">
          <v-checkbox
            v-model="rememberMe"
            label="Remember me"
            density="compact"
            hide-details
          />
          <v-btn variant="text" color="primary" class="text-caption" @click="needHelp">
            Need help?
          </v-btn>
        </v-row>
      </v-card-text>
      <v-card-actions class="d-flex flex-column">
        <v-btn variant="outlined" color="secondary" block @click="login">Login</v-btn>
        <v-btn color="primary" class="mb-2" block @click="register">Create Account</v-btn>
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
const rememberMe = ref(false)
const errorMessage = ref('')
const router = useRouter()
const API_BASE_URL = import.meta.env.VITE_BACKEND_URL
async function login() {
  errorMessage.value = ''

  if (!username.value || !password.value) {
    errorMessage.value = 'Please enter both username and password.'
    return
  }

  try {
    const response = await axios.post(`${API_BASE_URL}/api/login`, {
      username: username.value,
      password: password.value
    }, {
      withCredentials: true
    })

    if (rememberMe.value) {
      localStorage.setItem('rememberedUser', username.value)
    } else {
      localStorage.removeItem('rememberedUser')
    }

    router.push('/dashboard')

  } catch (err) {
    console.error(err)
    errorMessage.value = 'Login failed. Please check your credentials.'
  }
}

function register() {
  router.push(`/signup`)
}

function needHelp() {
  window.open('https://github.com/Programmierpraktikum-MVA/RobotDoc', '_blank')
}
</script>
