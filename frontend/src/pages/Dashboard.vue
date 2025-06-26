<template>
  <v-app>
    <Navbar />

    <v-main>
      <v-container class="mt-6">
        <h2 class="text-h5 font-weight-bold mb-4">Patient List</h2>
        <v-btn
          color="primary"
          class="mb-4"
          prepend-icon="mdi-plus"
          @click="router.push('/add')"
        >
          Create Patient
        </v-btn>

        <v-row>
          <v-col
            v-for="patient in patients"
            :key="patient.id"
            cols="12"
            sm="6"
            md="4"
          >
            <v-card color="surface" class="pa-4" elevation="4" hover>
              <div class="cursor-pointer" @click="$router.push(`/patient/${patient.id}`)">
                <div class="text-h6 font-weight-medium mb-2">{{ patient.name }}</div>

                <v-row dense class="mt-1">
                  <v-col cols="12" class="d-flex align-center mb-1">
                    <v-icon class="mr-2" size="20">mdi-cake-variant</v-icon>
                    {{ patient.age }} years
                  </v-col>

                  <v-col cols="12" class="d-flex align-center mb-1">
                    <v-icon class="mr-2" size="20">mdi-scale-bathroom</v-icon>
                    {{ patient.weight }} kg
                  </v-col>

                  <v-col cols="12" class="d-flex align-center mb-1">
                    <v-icon class="mr-2" size="20">mdi-gender-male-female</v-icon>
                    {{ patient.sex }}
                  </v-col>

                  <v-col cols="12" class="d-flex align-center">
                    <v-icon class="mr-2" size="20">mdi-heart-pulse</v-icon>
                    {{ patient.symptoms.filter(Boolean).join(', ') }}
                  </v-col>
                </v-row>
              </div>

              <v-card-actions class="mt-2 pa-0">
                <v-spacer />
                <v-btn
                  size="small"
                  variant="outlined"
                  color="primary"
                  @click.stop="editPatient(patient.id)"
                >
                  Edit
                </v-btn>
              </v-card-actions>
            </v-card>
          </v-col>
        </v-row>
      </v-container>
    </v-main>
  </v-app>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import axios from 'axios'
import Navbar from '@/components/Navbar.vue'

const router = useRouter()
const patients = ref([])
const API_BASE_URL = import.meta.env.VITE_BACKEND_URL
function editPatient(id) {
  router.push(`/edit/${id}`)
}


onMounted(async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/api/patients`, { withCredentials: true })
    console.log("Fetched patients:", response.data)
    patients.value = response.data
    console.log("Reactive patients:", patients.value)
  } catch (error) {
    console.error('Failed to fetch patients:', error)
  }
})

</script>
