import {createApp} from 'vue'
import App from './App.vue'
import '@varlet/ui/es/style.js'
import Snackbar from "@varlet/ui/es/snackbar";
import Dialog from "@varlet/ui/es/dialog";
import '@varlet/ui/es/dialog/style/index.js'
import '@varlet/ui/es/snackbar/style/index.js'
import '@arco-design/web-vue/dist/arco.css';

const app = createApp(App)
app.config.globalProperties.$tip = Snackbar
app.config.globalProperties.$dialog = Dialog
app.mount('#app')
