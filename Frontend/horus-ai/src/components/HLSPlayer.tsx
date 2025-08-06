"use client";
import { useEffect, useRef } from "react";
import Hls from "hls.js";

export default function HlsPlayer({ src }: { src: string }) {
  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    if (!videoRef.current) return;

    if (Hls.isSupported()) {
      const hls = new Hls();
      hls.loadSource(src);
      hls.attachMedia(videoRef.current);
    } else if (videoRef.current.canPlayType("application/vnd.apple.mpegurl")) { // jika browser safari
      videoRef.current.src = src;
    } else {
      console.error("HLS is not supported in this browser.");
    }
  }, [src]);

  return <video ref={videoRef} controls autoPlay className="w-full h-auto" />;
}
