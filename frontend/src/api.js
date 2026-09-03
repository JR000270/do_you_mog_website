//axios definition for calling api
import axios from 'axios';

//create instance of axios with the base URL
const api = axios.create({
    baseURL: "http://localhost:8000" //port/url the backend is running on
});

//export axios instance
export default api;