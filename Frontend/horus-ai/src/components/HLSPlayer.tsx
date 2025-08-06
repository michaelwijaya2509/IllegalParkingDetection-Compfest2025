"use client";
import { useEffect, useRef } from "react";
import Hls from "hls.js";

export default function HlsPlayer({ src }: { src: string }) {
  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    const video = videoRef.current;
    const m3u8Url = encodeURIComponent(src);
    const proxyUrl = `/api/stream?url=${m3u8Url}`;

    if (Hls.isSupported()) {
      const hls = new Hls();
      hls.loadSource(proxyUrl);
      hls.attachMedia(video!);
    } else if (video?.canPlayType("application/vnd.apple.mpegurl")) {
      video.src = proxyUrl;
    }
  }, [src]);

  return <video ref={videoRef} controls autoPlay className="w-full h-auto" />;
}
