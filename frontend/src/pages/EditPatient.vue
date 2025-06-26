<template>
  <v-app>
    <Navbar />
    <v-main>
      <v-container class="fill-height d-flex align-center justify-center">
        <v-card color="surface" class="pa-6" style="width: 100%; max-width: 900px;" elevation="6">
          <h2 class="text-h5 font-weight-bold mb-4 text-center">Edit Patient</h2>
          <v-form @submit.prevent="savePatient">
            <v-text-field v-model="patient.id" label="Patient ID" disabled />
            <v-text-field v-model="patient.name" label="Name" required />
            <v-text-field v-model="patient.age" label="Age" type="number" required />
            <v-text-field v-model="patient.weight" label="Weight (kg)" type="number" required />
            <v-select
              v-model="patient.sex"
              :items="['Male', 'Female']"
              label="Sex"
              required
            />
            <v-textarea
              v-model="symptomText"
              label="Symptoms (comma-separated)"
              rows="3"
              required
            />

            <v-btn color="primary" type="submit" class="mt-4" block>Save</v-btn>
            <v-btn color="error" class="mt-2" block @click="deletePatient">Delete</v-btn>
          </v-form>
        </v-card>
      </v-container>
    </v-main>
  </v-app>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import axios from 'axios'
import Navbar from '@/components/Navbar.vue'

const route = useRoute()
const router = useRouter()
const API_BASE_URL = import.meta.env.VITE_BACKEND_URL
const patient = ref({
  id: null,
  name: '',
  age: '',
  weight: '',
  sex: '',
  symptoms: []
})

const symptomText = ref('')

onMounted(async () => {
  const id = Number(route.params.id)
  try {
    const response = await axios.get(`${API_BASE_URL}/api/patient/${id}`)
    patient.value = response.data
    symptomText.value = Array.isArray(patient.value.symptoms)
      ? patient.value.symptoms.join(', ')
      : patient.value.symptoms
  } catch (error) {
    console.error('Failed to load patient:', error)
  }
})

async function savePatient() {
  // Convert comma-separated string to array
  patient.value.symptoms = symptomText.value
    .split(',')
    .map(s => s.trim())
    .filter(Boolean)

  try {
    await axios.put(`${API_BASE_URL}/api/patient/${patient.value.id}`, patient.value)
    alert('Patient saved!')
    router.push('/dashboard') // or wherever you list patients
  } catch (error) {
    console.error('Failed to save patient:', error)
    alert('Failed to save patient. See console for details.')
  }
}

async function deletePatient() {
  if (!confirm('Are you sure you want to delete this patient?')) return

  try {
    await axios.delete(`${API_BASE_URL}/api/patient/${patient.value.id}`, {withCredentials: true})
    alert('Patient deleted.')
    router.push('/dashboard')
  } catch (error) {
    console.error('Failed to delete patient:', error)
    alert('Failed to delete patient. See console for details.')
  }
}

</script>
