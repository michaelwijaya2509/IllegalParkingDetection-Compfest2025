/* eslint-disable react-hooks/exhaustive-deps */
import { PropagateLoader } from "react-spinners";
import { useMemo } from "react";

export const Loader = () => {
  const messages = [
    "Fetching data, please wait",
    "Almost there",
    "Loading HORUS experience",
    "Gathering information",
    "Preparing data",
  ];

  const randomMessage = useMemo(() => {
    return messages[Math.floor(Math.random() * messages.length)];
  }, []);

  return (
    <div className="fixed inset-0 text-white backdrop-opacity-80 backdrop-blur-lg backdrop-brightness-40 font-primary overflow-y-auto h-full w-full flex flex-col items-center justify-center z-[999] transition duration-300 ease-in-out">
      <p className="mb-10 text-white text-lg animate-pulse">{randomMessage}</p>
      <PropagateLoader color="#FFFFFF" />
    </div>
  );
};