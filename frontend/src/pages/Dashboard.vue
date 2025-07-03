<template>
  <v-app>
    <Navbar />

    <v-main>
      <v-container class="mt-6">
        <h2 class="text-h5 font-weight-bold mb-4">Patient List</h2>

        <!-- Loading Spinner -->
        <div v-if="loading" class="d-flex justify-center mt-8">
          <v-progress-circular indeterminate color="primary" size="50" />
        </div>

        <!-- Patient Cards -->
        <v-row v-else>
          <v-col
            v-for="patient in patients"
            :key="patient.id"
            cols="12"
            sm="6"
            md="4"
            lg="3"
          >
            <v-card
              color="surface"
              class="pa-4 text-center patient-card"
              elevation="4"
              hover
              :class="{ 'hovered-card': hoveredCardId === patient.id }"
              @mouseenter="hoveredCardId = patient.id"
              @mouseleave="hoveredCardId = null"
            >
              <div class="cursor-pointer" @click="$router.push(`/patient/${patient.id}`)">
                <div class="text-h6 font-weight-medium mb-4">{{ patient.name }}</div>

                <v-row dense justify="center" class="mb-2">
                  <v-col cols="12" class="d-flex justify-center mb-1">
                    <v-chip size="small" variant="elevated" color="primary">
                      <v-icon start size="18">mdi-cake-variant</v-icon>
                      {{ patient.age }} yrs
                    </v-chip>
                  </v-col>
                  <v-col cols="12" class="d-flex justify-center mb-1">
                    <v-chip size="small" variant="elevated" color="primary">
                      <v-icon start size="18">mdi-scale-bathroom</v-icon>
                      {{ patient.weight }} kg
                    </v-chip>
                  </v-col>
                  <v-col cols="12" class="d-flex justify-center mb-1">
                    <v-chip size="small" variant="elevated" color="primary">
                      <v-icon start size="18">mdi-gender-male-female</v-icon>
                      {{ patient.sex }}
                    </v-chip>
                  </v-col>
                  <v-col cols="12" class="d-flex justify-center mb-1">
                    <v-chip size="small" variant="outlined" color="primary">
                      <v-icon start size="18">mdi-heart-pulse</v-icon>
                      {{ patient.symptoms.filter(Boolean).join(', ') || 'No symptoms' }}
                    </v-chip>
                  </v-col>
                </v-row>
              </div>

              <div class="text-caption text-grey mb-2">ID: {{ patient.id }}</div>

              <v-card-actions class="justify-end pa-0">
                <v-btn
                  class="edit-button"
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

          <!-- Add Patient Card -->
          <v-col cols="12" sm="6" md="4" lg="3">
            <v-card
              class="pa-4 d-flex flex-column align-center justify-center add-card"
              color="surface"
              elevation="2"
              hover
              @click="router.push('/add')"
            >
              <v-icon size="48" color="primary">mdi-plus-circle-outline</v-icon>
              <div class="text-subtitle-1 mt-2 font-weight-medium text-primary">Add New Patient</div>
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
const loading = ref(true)
const hoveredCardId = ref(null)

const API_BASE_URL = import.meta.env.VITE_BACKEND_URL

function editPatient(id) {
  router.push(`/edit/${id}`)
}

onMounted(async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/api/patients`, { withCredentials: true })
    patients.value = response.data
  } catch (error) {
    console.error('Failed to fetch patients:', error)
  } finally {
    loading.value = false
  }
})
</script>

<style scoped>
.patient-card {
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.hovered-card {
  transform: scale(1.05);
  z-index: 2;
}
.edit-button {
  transition: all 0.3s ease;
  border: 1px solid var(--v-theme-primary);
  color: var(--v-theme-primary);
  background-color: transparent;
}
.hovered-card .edit-button:hover {
  background-color: var(--v-theme-primary);
  color: white;
}
.add-card {
  transition: transform 0.3s ease, background-color 0.3s ease;
  cursor: pointer;
}
.add-card:hover {
  transform: scale(1.05);
  background-color: var(--v-theme-surface-variant);
}
</style>
