<template>
  <v-app>
    <Navbar />
    <v-main>
      <v-container class="fill-height d-flex align-center justify-center">
        <v-card color="surface" class="pa-6" style="width: 100%; max-width: 900px;" elevation="6">
          <h2 class="text-h5 font-weight-bold mb-4 text-center">Edit Patient</h2>
          <v-form @submit.prevent="savePatient">
            <v-text-field v-model="patient.id" label="Patient ID" disabled rounded variant="outlined" />
            <v-text-field v-model="patient.name" label="Name" required rounded variant="outlined" />
            <v-text-field v-model="patient.age" label="Age" type="number" required rounded variant="outlined" />
            <v-text-field v-model="patient.weight" label="Weight (kg)" type="number" required rounded variant="outlined" />
            <v-select
              v-model="patient.sex"
              :items="['Male', 'Female']"
              label="Sex"
              required
              rounded
              variant="outlined"
            />
            <v-textarea
              v-model="symptomText"
              label="Symptoms (comma-separated)"
              rows="3"
              required
              rounded
              variant="outlined"
            />

            <!-- Combined Upload Image Block -->
            <v-card class="pa-4 rounded-lg mb-6 mt-4" color="surface" elevation="2">
              <v-file-input
                v-model="uploadedImage"
                label="Upload New Image"
                accept="image/*"
                prepend-icon="mdi-camera"
                show-size
                hide-details
                density="comfortable"
                rounded
                variant="outlined"
              />
              <v-btn
                color="primary"
                class="mt-3"
                block
                prepend-icon="mdi-upload"
                @click="uploadImage"
              >
                Add Image
              </v-btn>
            </v-card>

            <!-- Action Buttons -->
            <v-btn
              color="primary"
              class="mb-3"
              block
              prepend-icon="mdi-content-save"
              @click="savePatient"
            >
              Save
            </v-btn>
            <v-btn
              color="error"
              block
              prepend-icon="mdi-delete"
              @click="deletePatient"
            >
              Delete
            </v-btn>
          </v-form>

          <!-- Display Existing Images -->
          <div class="mt-6">
            <h3 class="text-subtitle-1 font-weight-bold mb-2">Current Images</h3>
            <v-row>
              <v-col
                v-for="(img, index) in patientImages"
                :key="index"
                cols="12" sm="6" md="4" lg="3"
              >
                <v-card class="pa-0 rounded-lg overflow-hidden position-relative" elevation="2">
                  <v-img :src="img.url" class="rounded" contain max-height="200" />
                  <v-btn icon small class="position-absolute top-0 right-0 ma-1" color="error" @click="deleteImage(img.id)">
                    <v-icon>mdi-close</v-icon>
                  </v-btn>
                </v-card>
              </v-col>
            </v-row>
          </div>
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
const uploadedImage = ref(null)
const patientImages = ref([])

onMounted(async () => {
  const id = Number(route.params.id)
  try {
    const response = await axios.get(`${API_BASE_URL}/api/patient/${id}`)
    patient.value = response.data
    symptomText.value = Array.isArray(patient.value.symptoms)
      ? patient.value.symptoms.join(', ')
      : patient.value.symptoms
    await fetchImages()
  } catch (error) {
    console.error('Failed to load patient:', error)
  }
})

async function fetchImages() {
  try {
    const imgResponse = await axios.get(`${API_BASE_URL}/api/patient/${patient.value.id}/images`)
    patientImages.value = imgResponse.data.map(img => ({
      id: img.url.split('/').pop(),
      url: `${API_BASE_URL}${img.url}`
    }))
  } catch (err) {
    console.error('Failed to load images:', err)
  }
}

async function uploadImage() {
  if (!uploadedImage.value) return

  const formData = new FormData()
  formData.append('image', uploadedImage.value)

  try {
    await axios.post(`${API_BASE_URL}/api/patient/${patient.value.id}/images`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      withCredentials: true
    })
    uploadedImage.value = null
    await fetchImages()
  } catch (err) {
    console.error('Upload failed:', err)
  }
}

async function deleteImage(imageId) {
  try {
    await axios.delete(`${API_BASE_URL}/api/image/${imageId}`, { withCredentials: true })
    await fetchImages()
  } catch (err) {
    console.error('Delete failed:', err)
  }
}

async function savePatient() {
  patient.value.symptoms = symptomText.value
    .split(',')
    .map(s => s.trim())
    .filter(Boolean)

  try {
    await axios.put(`${API_BASE_URL}/api/patient/${patient.value.id}`, patient.value)
    router.push('/dashboard')
  } catch (error) {
    console.error('Failed to save patient:', error)
  }
}

async function deletePatient() {
  try {
    await axios.delete(`${API_BASE_URL}/api/patient/${patient.value.id}`, { withCredentials: true })
    router.push('/dashboard')
  } catch (error) {
    console.error('Failed to delete patient:', error)
  }
}
</script>