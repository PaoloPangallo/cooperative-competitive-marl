import api from "./client";

export const startTraining = async () => {
  const res = await api.post("/train/mappo");
  return res.data;
};

export const getTrainStatus = async () => {
  const res = await api.get("/train/status");
  return res.data;
};

export const stopTraining = async () => {
  const res = await api.post("/train/stop");
  return res.data;
};
