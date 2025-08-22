import { PropagateLoader } from "react-spinners";

export const Loader = () => {
  return (
    <div className="fixed inset-0 backdrop-opacity-80 backdrop-blur-lg backdrop-brightness-40 font-primary overflow-y-auto h-full w-full flex items-center justify-center z-999 transition duration-300 ease-in-out">
      <PropagateLoader color="#FFFF" />
    </div>
  );
};
