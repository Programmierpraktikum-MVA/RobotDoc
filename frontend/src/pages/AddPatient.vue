<template>
  <v-app>
    <Navbar />
    <v-main>
      <v-container class="fill-height d-flex align-center justify-center">
        <v-card color="surface" class="pa-6" style="width: 100%; max-width: 900px;" elevation="6">
          <h2 class="text-h5 font-weight-bold mb-4 text-center">Add New Patient</h2>
          <v-form @submit.prevent="submitForm">
            <v-text-field v-model="patient.name" label="Name" required />
            <v-text-field v-model.number="patient.age" label="Age" type="number" required />
            <v-text-field v-model.number="patient.weight" label="Weight (kg)" type="number" required />
            <v-select
              v-model="patient.sex"
              :items="['Male', 'Female']"
              label="Sex"
              required
            />
            <v-textarea
              v-model="patient.symptoms"
              label="Symptoms (comma separated)"
              rows="3"
              required
            />
            <v-btn color="primary" type="submit" class="mt-4" block>Add Patient</v-btn>
          </v-form>
        </v-card>
      </v-container>
    </v-main>
  </v-app>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import axios from 'axios'
import Navbar from '@/components/Navbar.vue'

const router = useRouter()
const API_BASE_URL = import.meta.env.VITE_BACKEND_URL
const patient = ref({
  name: '',
  age: '',
  weight: '',
  sex: '',
  symptoms: ''
})

async function submitForm() {
  try {
    await axios.post(`${API_BASE_URL}/api/createPatient`, {
      name: patient.value.name,
      age: patient.value.age,
      weight: patient.value.weight,
      sex: patient.value.sex,
      symptoms: patient.value.symptoms.split(',').map(s => s.trim())
    }, { withCredentials: true })

    alert('Patient added successfully!')
    router.push('/dashboard')
  } catch (err) {
    console.error(err)
    alert('Error adding patient.')
  }
}
</script>
