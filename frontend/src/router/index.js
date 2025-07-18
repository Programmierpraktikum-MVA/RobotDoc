import { createRouter, createWebHistory } from 'vue-router'
import Login from '../pages/Login.vue'
import Dashboard from '../pages/Dashboard.vue'
import Patient from '../pages/Patient.vue'
import EditPatient from '../pages/EditPatient.vue'
import Signup from '../pages/Signup.vue'

const routes = [
  { path: '/', redirect: '/login' },
  { path: '/login', component: Login, meta: {title: 'Login (RobotDoc)'} },
  { path: '/signup', component: Signup, meta: {title: 'Sign Up (RobotDoc)'} },
  {
    path: '/dashboard',
    component: Dashboard,
    meta: { requiresAuth: true, title: 'Dashboard (RobotDoc)' }
  },
  {
    path: '/patient/:id',
    component: Patient,
    meta: { requiresAuth: true, title: 'View Patient (RobotDoc)' }
  },
  {
    path: '/add',
    component: AddPatient,
    meta: { requiresAuth: true, title: 'Add Patient (RobotDoc)' }
  },
  {
    path: '/edit/:id',
    component: EditPatient,
    meta: { requiresAuth: true, title: 'Edit Patient (RobotDoc)' }
  }
]


const router = createRouter({
  history: createWebHistory(),
  routes
})

import axios from 'axios'
import AddPatient from '@/pages/AddPatient.vue'
const API_BASE_URL = import.meta.env.VITE_BACKEND_URL


router.beforeEach(async (to, from, next) => {
  const defaultTitle = 'RobotDoc'
  document.title = to.meta.title || defaultTitle
  if (to.meta.requiresAuth) {
    try {
      const res = await axios.get(`${API_BASE_URL}/api/check_session`, {
        withCredentials: true
      })
      if (res.status === 200) {
        next()
      } else {
        next('/login')
      }
    } catch (e) {
      next('/login')
    }
  } else {
    next()
  }
  
})

export default router
