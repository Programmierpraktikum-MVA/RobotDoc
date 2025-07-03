<template>
  <v-app-bar color="surface" flat app class="px-4">
    <!-- Logo & Title -->
    <div class="d-flex align-center">
      <v-icon color="primary" class="mr-2">mdi-robot</v-icon>
      <v-toolbar-title class="font-weight-bold text-h6">RobotDoc</v-toolbar-title>
    </div>

    <!-- Navigation -->
    <v-btn variant="text" class="ml-4" @click="$router.push('/dashboard')" color="primary">
      Patients
    </v-btn>

    <v-spacer />

    <!-- Logout Icon Button -->
    <v-btn icon @click="logout" title="Logout">
      <v-icon>mdi-logout</v-icon>
    </v-btn>
  </v-app-bar>
</template>

<script setup>
import axios from 'axios'
import { useRouter } from 'vue-router'

const router = useRouter()
const API_BASE_URL = import.meta.env.VITE_BACKEND_URL

async function logout() {
  try {
    await axios.post(`${API_BASE_URL}/api/logout`, {}, { withCredentials: true })
  } catch (err) {
    console.warn('Logout failed on server, continuing client-side logout.')
  }

  localStorage.removeItem('rememberedUser')
  router.push('/login')
}
</script>
