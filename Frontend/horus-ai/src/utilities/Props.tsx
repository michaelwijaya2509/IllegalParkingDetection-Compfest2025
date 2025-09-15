import { ToastOptions } from "react-toastify";
import { Slide } from "react-toastify/unstyled";

export const ToastProps: ToastOptions = {
  position: "top-center",
  autoClose: 3000,
  hideProgressBar: true,
  closeButton: false,
  closeOnClick: true,
  pauseOnHover: true,
  draggable: true,
  theme: "dark",
  transition: Slide,
}