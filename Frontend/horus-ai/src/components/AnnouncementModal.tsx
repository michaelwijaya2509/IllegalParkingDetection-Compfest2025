import { useRouter } from "next/navigation";
import { JSX} from "react";
import { motion } from "framer-motion";

interface ModalProps {
  title: string;
  buttonLabel: string;
  isLoading: boolean;
  descriptions: JSX.Element[];
  setState: () => void;
  setOnBackgroundClick?: () => void;
}

const AnnouncementModal: React.FC<ModalProps> = ({
  title,
  buttonLabel,
  descriptions,
  setState,
  setOnBackgroundClick,
}) => {
  const navigate = useRouter();

  const handleClick = () => {
    setState();
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
          {
            descriptions.map((description, index) => (
              <div key={index} className={`mt-2 text-sm text-gray-300 ${index === 0 ? '' : 'mt-4'}`}>
                {description}
              </div>
            ))
          }
          <div className="flex flex-col justify-center mt-4 items-center">
            <button
              onClick={handleClick}
              className="w-full text-center py-2 px-4 bg-yellow-500/20 hover:bg-yellow-500/40 
                         text-yellow-400 border border-yellow-500/30 rounded-lg transition-all 
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

export default AnnouncementModal;
