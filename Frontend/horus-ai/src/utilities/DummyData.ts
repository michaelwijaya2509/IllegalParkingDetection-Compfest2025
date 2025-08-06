import { CCTVData, IllegalParkingData } from "./Types";

  export const illegalParkingLocations: IllegalParkingData[] = [
    {
      coordinates: [-6.920286695868032, 107.59829340956341],
      narration: "Sebuah mobil terparkir di tepi jalan depan toko. Mobil tersebut menghalangi jalur akses ke toko dan mengganggu pejalan kaki.",
      videoUrl: "https://drive.google.com/file/d/1_E6eFwZ__Ifb-jTM-S0aDajJimLCBaLl/preview",
      locationName: "Jl. Gardujati",
      timestamp: '2023-10-01T12:00:00Z',
      urgency: 2
    },
    {
      coordinates: [-6.920589141145961, 107.59833250494219],
      narration: "Mobil terparkir sembarangan di depan toko, menghalangi akses masuk dan keluar kendaraan lain.",
      videoUrl: "https://drive.google.com/file/d/19zIremy7wi1H1zjU7xAbOm16RrHqQT_8/preview",
      locationName: "Jl. Gardujati",
      timestamp: '2023-10-02T12:35:20Z',
      urgency: 3
    }
  ];

  export const cctvLocations: CCTVData[] = [
    {
      coordinates: [-6.920065458288789, 107.59843008554336],
      streamUrl: "https://pelindung.bandung.go.id:3443/video/DISHUB/spgardujati.m3u8",
      description: "CCTV Jl. Gardujati"
    }
  ];