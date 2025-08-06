import { NextRequest } from "next/server";

export async function GET(req: NextRequest) {
  const target = req.nextUrl.searchParams.get("url");
  if (!target) {
    return new Response("Missing 'url' query parameter", { status: 400 });
  }

  try {
    const upstream = await fetch(target);
    const contentType = upstream.headers.get("content-type") || "application/octet-stream";

    if (contentType.includes("application/vnd.apple.mpegurl") || target.endsWith(".m3u8")) {
      let playlist = await upstream.text();

      const baseUrl = target.substring(0, target.lastIndexOf("/") + 1);

      playlist = playlist.split("\n").map(line => {
        if (line && !line.startsWith("#")) {
          const absoluteUrl = new URL(line, baseUrl).toString();
          return `/api/stream?url=${encodeURIComponent(absoluteUrl)}`;
        }
        return line;
      }).join("\n");

      return new Response(playlist, {
        status: upstream.status,
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "GET, OPTIONS",
          "Access-Control-Allow-Headers": "*",
          "Content-Type": contentType,
        },
      });
    }

    return new Response(upstream.body, {
      status: upstream.status,
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Content-Type": contentType,
      },
    });
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  } catch (err) {
    return new Response("Failed to fetch target URL", { status: 500 });
  }
}
