import { useRouter } from "next/navigation";
import { useContext, useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Listbox, ListboxButton, ListboxOption, ListboxOptions } from "@headlessui/react";
import { DataContext } from "@/context/Context";

interface ModalProps {
  title: string;
  buttonLabel: string;
  isLoading: boolean;
  cameraIds: string[];
  setState: (cameraId: string | null) => void;
  setOnBackgroundClick?: () => void;
}

const FilterModal: React.FC<ModalProps> = ({
  title,
  buttonLabel,
  cameraIds,
  isLoading,
  setState,
  setOnBackgroundClick,
}) => {
  const navigate = useRouter();
  const [selectedCameraId, setSelectedCameraId] = useState<string>("");

  const context = useContext(DataContext);
  const { selectedCamId } = context || { selectedCamId: null };

  useEffect(() => {
    if (selectedCamId) {
      setSelectedCameraId(selectedCamId);
    }
  }, [])

  const handleClick = () => {
    setState(selectedCameraId === "All Cameras" ? null : selectedCameraId);
  }

  return (
    <div
      className="fixed inset-0 backdrop-opacity-80 backdrop-blur-lg backdrop-brightness-40 font-primary 
                 overflow-y-auto h-full w-full flex items-center justify-center z-50 transition duration-300 ease-in-out"
      onClick={setOnBackgroundClick ? setOnBackgroundClick : () => navigate.back()}
    >
      <motion.div
        className="py-8 px-10 w-96 shadow-xl rounded-lg border border-gray-700 bg-tile1/100 hover:border-gray-600"
        onClick={(e) => e.stopPropagation()}
        initial={{ opacity: 0, y: 100 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -100 }}
      >
        <div className="text-center">
          <h3 className="text-2xl font-bold text-secondary">{title}</h3>

          <p className="mt-2 text-sm text-gray-300">
            Please select a camera from the list below
          </p>

          <div className="mt-4">
            <Listbox
              value={selectedCameraId}
              onChange={(val) => setSelectedCameraId(val)}
            >
              <div className="relative text-black">
                <ListboxButton
                  className="w-full px-4 py-2 text-left bg-white border rounded-lg 
                             cursor-pointer font-primary"
                >
                  {selectedCameraId || "Select a Camera"}
                </ListboxButton>
                <ListboxOptions
                  className="absolute mt-2 w-full bg-white border rounded-lg shadow-lg 
                             max-h-60 overflow-auto font-primary z-50 text-left"
                >
                  {["All Cameras", ...cameraIds].map((id, index) => (
                    <ListboxOption
                      key={id + index.toString()}
                      value={id}
                      className={({ active }) =>
                        `px-4 py-2 cursor-pointer ${
                          active
                            ? "bg-gray-200"
                            : "text-black hover:bg-gray-100"
                        }`
                      }
                    >
                      {id}
                    </ListboxOption>
                  ))}
                </ListboxOptions>
              </div>
            </Listbox>
          </div>

          <div className="flex flex-col justify-center mt-4 items-center">
            <button
              onClick={handleClick}
              disabled={isLoading || !selectedCameraId}
              className="w-full text-center py-2 px-4 bg-green-500/20 hover:bg-green-500/40 
                         text-green-400 border border-green-500/30 rounded-lg transition-all 
                         duration-200 font-semibold hover:cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {buttonLabel}
            </button>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default FilterModal;
