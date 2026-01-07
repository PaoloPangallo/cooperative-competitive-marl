import { api } from "./client";

export const getLatestTrajectory = () =>
  api.get("/train/trajectory");

export const listTrajectories = () =>
  api.get("/train/trajectory/list");

export const getTrajectoryByIter = (iter) =>
  api.get(`/train/trajectory/${iter}`);
