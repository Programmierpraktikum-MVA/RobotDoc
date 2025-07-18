<template>
  <v-app>
    <Navbar />
    <v-main>
      <v-container class="mt-6">
        <v-row>
          <!-- Patient Profile -->
          <v-col cols="12" md="4">
            <v-card color="surface" class="pa-5" elevation="4" style="max-width: 480px; margin: auto;">
              <h2 class="text-h6 font-weight-bold text-center mb-4">Patient Profile</h2>

              <div v-if="loadingDetails" class="d-flex justify-center align-center" style="height: 200px;">
                <v-progress-circular indeterminate color="primary" size="40" />
              </div>

              <template v-else>
                <div class="d-flex flex-column align-center text-center mb-4">
                  <v-avatar size="72" class="mb-2" color="primary" variant="tonal">
                    <v-icon size="36">mdi-account</v-icon>
                  </v-avatar>
                  <div class="text-h6 font-weight-bold">{{ patient.name }}</div>
                  <div class="text-caption text-medium-emphasis">ID: {{ patient.id }}</div>
                </div>

                <v-divider class="my-2" />

                <v-row dense>
                  <v-col cols="6" class="text-left">
                    <div class="text-subtitle-2 font-weight-bold">Age</div>
                    <div class="text-body-2 text-medium-emphasis">{{ patient.age }}</div>
                  </v-col>
                  <v-col cols="6" class="text-left">
                    <div class="text-subtitle-2 font-weight-bold">Sex</div>
                    <div class="text-body-2 text-medium-emphasis">{{ patient.sex }}</div>
                  </v-col>
                  <v-col cols="6" class="text-left">
                    <div class="text-subtitle-2 font-weight-bold">Weight</div>
                    <div class="text-body-2 text-medium-emphasis">{{ patient.weight }} kg</div>
                  </v-col>
                  <v-col cols="12" class="text-left">
                    <div class="text-subtitle-2 font-weight-bold mb-1">Symptoms</div>
                    <div class="d-flex flex-wrap gap-2">
                      <v-chip
                        v-for="(symptom, i) in (Array.isArray(patient.symptoms) ? patient.symptoms.filter(Boolean) : [])"
                        :key="i"
                        size="small"
                        variant="outlined"
                        color="primary"
                      >
                        {{ symptom }}
                      </v-chip>
                      <span v-if="!Array.isArray(patient.symptoms)">{{ patient.symptoms }}</span>
                    </div>
                  </v-col>
                </v-row>
              </template>
            </v-card>
          </v-col>

          <!-- Chat -->
          <v-col cols="12" md="8">
            <ChatWindow :patient-id="patient.id" />
          </v-col>

          <!-- Images (only shown if available) -->
          <v-col cols="12" v-if="!loadingImages && patientImages.length > 0">
            <v-card color="surface" class="pa-4 mt-4" elevation="4" min-height="200px">
              <h3 class="text-h6 font-weight-bold mb-4">Patient Images</h3>

              <v-row>
                <v-col
                  v-for="(image, index) in patientImages"
                  :key="index"
                  cols="12" sm="6" md="4" lg="3"
                >
                  <v-img
                    :src="image.url"
                    class="rounded cursor-pointer"
                    contain
                    @click="() => openImage(image.url)"
                  />
                </v-col>
              </v-row>
            </v-card>
          </v-col>

          <!-- Loading Spinner for Images -->
          <v-col cols="12" v-else-if="loadingImages">
            <v-card color="surface" class="pa-4 mt-4" elevation="4" min-height="200px">
              <div class="d-flex justify-center align-center" style="height: 150px;">
                <v-progress-circular indeterminate color="primary" size="40" />
              </div>
            </v-card>
          </v-col>
        </v-row>

        <!-- Fullscreen Image Dialog -->
        <v-dialog v-model="dialog" max-width="90vw" persistent>
          <v-card color="surface" class="pa-2">
            <v-img :src="selectedImage" max-height="80vh" class="mx-auto" contain />
            <v-card-actions>
              <v-spacer />
              <v-btn color="primary" @click="dialog = false">Close</v-btn>
            </v-card-actions>
          </v-card>
        </v-dialog>
      </v-container>
    </v-main>
  </v-app>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import axios from 'axios'
import Navbar from '@/components/Navbar.vue'
import ChatWindow from '@/components/ChatWindow.vue'

const route = useRoute()

const patient = ref({
  id: null,
  name: '',
  age: '',
  weight: '',
  sex: '',
  symptoms: []
})

const patientImages = ref([])
const selectedImage = ref(null)
const dialog = ref(false)

const loadingDetails = ref(true)
const loadingImages = ref(true)

const API_BASE_URL = import.meta.env.VITE_BACKEND_URL

function openImage(url) {
  selectedImage.value = url
  dialog.value = true
}

onMounted(async () => {
  const id = Number(route.params.id)

  try {
    const response = await axios.get(`${API_BASE_URL}/api/patient/${id}`)
    patient.value = response.data
  } catch (error) {
    console.error('Failed to fetch patient details:', error)
  } finally {
    loadingDetails.value = false
  }

  try {
    const imgResponse = await axios.get(`${API_BASE_URL}/api/patient/${id}/images`)
    patientImages.value = imgResponse.data.map(img => ({
      url: `${API_BASE_URL}${img.url}`
    }))
  } catch (error) {
    console.error('Failed to fetch patient images:', error)
  } finally {
    loadingImages.value = false
  }
})
</script>
