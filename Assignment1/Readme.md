# Assignment 1: Semantic Landmark Extraction and Classification

**Student:** Varad Jahagirdar (1233869093)

**Course:** RAS 598 — Mobile Robotics  

**Robot:** OAK-D Camera (Simulation via ROS 2 Bags)  

**ROS Distro:** Jazzy  

---

## Overview

This assignment implements a full perception pipeline that processes raw RGB-D point clouds from an OAK-D camera and detects, localizes, and semantically labels cylinders in the environment. The pipeline runs entirely on recorded ROS 2 bags without a physical robot.

The node subscribes to `/oakd/points` (`sensor_msgs/PointCloud2`), processes each frame through a multi-stage pipeline, and publishes colored cylinder markers to RViz along with intermediate point clouds for debugging at each stage.

---

## Pipeline Summary

```
Raw PointCloud2
      │
      ▼
 [Box Filter]           — Remove points outside XYZ region of interest
      │
      ▼
 [Voxel Downsample]     — Keep one point per 3cm voxel (np.unique)
      │
      ▼
 [Normal Estimation]    — SVD of k-neighbor patch → surface normal per point
      │
      ▼
 [Plane RANSAC × 3]     — Remove floor, table, ceiling iteratively
      │
      ▼
 [Euclidean Clustering] — BFS + cKDTree radius search → separate objects
      │
      ▼
 [Cylinder RANSAC]      — cross(n1, n2) → axis → perpendicular distance inliers
      │
      ▼
 [HSV Classification]   — mean inlier RGB → HSV → hue range → Red/Green/Blue
      │
      ▼
 [Deduplication]        — Suppress duplicate markers within 0.20 m
      │
      ▼
 [RViz MarkerArray]     — Colored cylinder markers at detected positions
```

---

## File Structure

```
Assignment_1/
├── cylinder_pipeline.py   — Main ROS 2 node (full pipeline implementation)
├── cylinders.rviz         — RViz configuration file
└── README.md              — This file
```

---

## Dependencies

- ROS 2 Jazzy
- Python 3
- NumPy
- SciPy (`cKDTree` only — for neighbor search)
- `sensor_msgs`, `visualization_msgs`, `geometry_msgs` (standard ROS 2 packages)

---

## How to Run

**Step 1 — Set simulation environment:**
```bash
set-ros-env sim
```
> Close all terminals and reopen after this step.

**Step 2 — Play a bag file at reduced rate:**
```bash
ros2 bag play <your_bag_file>.mcap --loop --rate 0.5
```

**Step 3 — Run the pipeline node:**
```bash
python3 cylinder_pipeline.py
```

**Step 4 — Launch RViz with saved config:**
```bash
rviz2 -d cylinders.rviz
```

**RViz topics to subscribe to:**

| Topic | Type | Purpose |
|---|---|---|
| `pipeline/stage0_box` | PointCloud2 | After box filter + voxel downsample |
| `pipeline/stage1_no_planes` | PointCloud2 | After plane removal |
| `pipeline/stage3_candidates` | PointCloud2 | Clusters (each a different debug color) |
| `viz/detections` | MarkerArray | Final labeled cylinder markers |

---

## Algorithm Details

### Task 0 — Preprocessing

**Box Filter:** A single boolean mask keeps only points within the configured XYZ bounds. This is O(N) with no loops — one vectorized NumPy operation.

**Voxel Downsampling:** Each point's coordinates are divided by `voxel_size` and floored to produce integer voxel indices. `np.unique(axis=0)` with `return_index=True` finds the first point in each unique voxel, reducing the cloud from ~100k points to ~3–5k. After testing `0.01` (too dense, processing lag), `0.02` (good detail), and `0.03` (optimal speed/detail balance), `0.03` was selected as the final value.

---

### Task 1 — Normal Estimation + Plane Segmentation (RANSAC)

**Normal Estimation:** For each point, the k nearest neighbors are found with `cKDTree`. SVD is applied to the centered (k × 3) neighborhood matrix. The last row of `Vt` — the direction of minimum variance — is the surface normal perpendicular to the local surface patch.

**Plane RANSAC:** Each iteration samples 3 random points, computes the plane normal via cross product, checks alignment with the expected floor normal using a dot product threshold (`normal_thresh = 0.85`), and counts inliers as points within `floor_dist` of the plane. The best plane is removed and the process repeats up to `num_plane_removals = 3` times to iteratively strip the floor, table, and ceiling.

---

### Task 2 — Euclidean Clustering

BFS with a `cKDTree` radius query groups spatially connected points into clusters. Each unvisited point starts a new cluster. All neighbors within `cluster_radius` are added to the queue and marked visited. Clusters outside `[cluster_min_pts, cluster_max_pts]` are discarded. A debug publisher colors each cluster distinctly in RViz for visual inspection.

---

### Task 3 — Cylinder Detection (RANSAC)

Two points and their surface normals are sampled. On a cylinder surface, normals point radially outward — perpendicular to the axis. The cross product of two radial normals gives the axis direction. The axis is verified to be approximately vertical using `cyl_axis_thresh`. Inliers are counted as points whose perpendicular distance to the axis is within `cyl_inlier_tol` of `cyl_radius`. All inlier distance calculations are fully vectorized with NumPy — no inner loops.

A **deduplication step** runs after all clusters are processed: if two detected cylinders are within `0.20 m` of each other on the XZ plane, the duplicate is suppressed. This prevents one physical cylinder from generating two overlapping markers when sparse point coverage causes it to split into nearby clusters.

---

### Task 4 — Semantic Labeling via HSV

The average RGB color of cylinder inlier points (not the full cluster — only RANSAC surface inliers) is converted to HSV using a pure Python implementation. Hue thresholds were derived from actual terminal HSV debug output collected across multiple frames:

| Color | Hue Range | Saturation Guard | Measured HSV (terminal) |
|---|---|---|---|
| Red | h < 15° or h > 320° | s ≥ 0.10 | h=335–342, s=0.17–0.23, v=0.65–0.82 |
| Green | 90° ≤ h ≤ 150° | s ≥ 0.10 | h=91–100, s=0.17–0.25, v=0.53–0.63 |
| Blue | 200° ≤ h ≤ 270° | s ≥ 0.10 | h=210–221, s=0.48–0.70, v=0.39–0.44 |

A saturation guard (`s < 0.10`) and value guard (`v < 0.15`) prevent misclassifying grey noise or dark unlit surfaces.

---

## Configuration Parameters

All parameters are in `PipelineConfig` at the top of `cylinder_pipeline.py`:

| Parameter | Value | Description |
|---|---|---|
| `voxel_size` | `0.03` | Voxel grid cell size in meters |
| `floor_dist` | `0.04` | Plane inlier threshold in meters |
| `normal_thresh` | `0.85` | Min dot product with vertical for floor planes |
| `num_plane_removals` | `3` | Number of planes to strip per frame |
| `cluster_radius` | `0.06` | BFS neighbor search radius in meters |
| `cluster_min_pts` | `20` | Minimum points per valid cluster |
| `cyl_radius` | `0.055` | Expected cylinder radius in meters |
| `cyl_inlier_tol` | `0.030` | Tolerance around expected radius |
| `cyl_axis_thresh` | `0.70` | Min dot product with vertical for cylinder axis |
| `cyl_min_inliers` | `10` | Minimum inliers to accept a detection |
| `max_cylinders` | `3` | Maximum cylinders to detect per frame |
| `marker_alpha` | `0.9` | Cylinder marker opacity in RViz |

---

## RViz Configuration

The saved `cylinders.rviz` configuration includes the following customizations made during debugging:

- **Background color:** Dark grey `(48, 48, 48)` — better contrast against white point clouds
- **PointCloud2 point size:** `0.03 m`, style: `Flat Squares` — larger points easier to inspect per-stage
- **Fixed Frame:** `oakd_rgb_camera_optical_frame`
- **Marker opacity:** `0.9` — increased from default `0.8` for better visibility in dense scenes

---

## Library Compliance

- **Allowed and used:** NumPy, `scipy.spatial.cKDTree` (neighbor search only), standard ROS 2 packages
- **Not used:** Open3D, PCL, scikit-learn, SciPy RANSAC, SciPy clustering
- All geometric logic — RANSAC, normal estimation, clustering, distance calculations — is implemented with pure NumPy

---

## Debugging and Tuning

Getting the pipeline working correctly required several rounds of parameter tuning. Each issue was diagnosed by adding HSV debug log output and inspecting intermediate point clouds in RViz at each pipeline stage.

### 1. Cylinder Marker Orientation

The identity quaternion (`orientation.w = 1.0`, all others `0`) produces the correct upright cylinder orientation in the `oakd_rgb_camera_optical_frame`. The RViz cylinder marker's default Z axis aligns correctly with vertical in this camera frame without any additional rotation.

### 2. Voxel Size Tuning

Tested `0.01`, `0.02`, and `0.03`. At `0.01` the node could not keep up with 0.5x playback — processing time exceeded the message interval. At `0.03` the point count dropped to ~3–5k per frame, processing in real time while retaining sufficient detail for cylinder fitting.

### 3. Cluster Radius Tuning for `rgbd_bag_2`

With three cylinders in the scene, `cluster_radius` needed careful tuning:

| Value | Problem |
|---|---|
| `0.08` | Too large — merged nearby cylinders into one blob |
| `0.05` | Too small — split each cylinder into multiple fragments |
| `0.06` | Correct — consistently produced 3–4 distinct clusters ✓ |

The right value was found by printing cluster sizes and centers at each frame and narrowing between the two failure modes.

### 4. HSV Threshold Derivation for Red Cylinder

Color thresholds were not guessed — they were derived from actual HSV debug log output printed by the node for every detected cylinder:

```
HSV debug: h=336.2  s=0.216  v=0.817  → label=red
HSV debug: h=96.9   s=0.196  v=0.573  → label=green
HSV debug: h=216.0  s=0.657  v=0.413  → label=blue
```

The red cylinder consistently read at `h=335–342°` with low saturation `s=0.17–0.23`. Two fixes were applied:
- Lowered saturation guard from `s < 0.20` to `s < 0.10` to allow washed-out colors through
- Widened the red upper boundary to `h > 320°` to capture hues in the magenta-red range

### 5. Floor Distance Tuning

Increasing `floor_dist` from `0.02` to `0.04` improved floor removal on uneven lab surfaces, leaving fewer residual floor points after the plane stripping stage that were previously confusing the clustering step.

### 6. Duplicate Detection Suppression

With `cluster_min_pts` lowered to `20` to handle sparse clusters, some cylinders split into two nearby fragments that each passed RANSAC independently. A post-detection deduplication step was added: any two detections within `0.20 m` of each other on the XZ plane are merged — the first detection (larger cluster) is kept and the second is suppressed.

---

## Results

### `rgbd_bag_0` — Single Green Cylinder

Static scene with one green cylinder. The pipeline correctly detects and labels it across all frames with consistent position output in the terminal.

<!-- ADD SCREENSHOT: RViz showing green cylinder marker with stage0 point cloud -->


<img width="1661" height="814" alt="image" src="https://github.com/user-attachments/assets/a20d7f69-75c5-4bf2-a8b2-5e155bd69912" />
<img width="1303" height="640" alt="image" src="https://github.com/user-attachments/assets/a960d9ec-3a16-4402-a1dd-29c1db78fe56" />

---

### `rgbd_bag_1` — Robot Moving Around Cylinders

The robot moves around the scene, continuously changing viewpoint. The pipeline detects and tracks the cylinder reliably across all frames despite changing depth and viewing angle.

<!-- ADD SCREENSHOT: RViz showing detection during motion -->
<!-- ADD SCREEN RECORDING: rgbd_bag_1 tracking across frames -->

<img width="1663" height="769" alt="image" src="https://github.com/user-attachments/assets/18700185-6c94-4430-9f24-6fa345ba50d7" />
<img width="1304" height="640" alt="image" src="https://github.com/user-attachments/assets/7478da1d-5e0a-468a-82e1-c4f1ac1b1210" />


### `rgbd_bag_2` — Three Cylinders (Red, Green, Blue)

Three cylinders at different positions in the scene. All three are detected and correctly labeled by color in the same frame.

<!-- ADD SCREENSHOT: RViz showing all 3 labeled cylinders simultaneously -->
<!-- ADD SCREEN RECORDING: rgbd_bag_2 showing stable multi-cylinder detection -->

<img width="1662" height="768" alt="image" src="https://github.com/user-attachments/assets/23ac4ff3-a530-4ad3-bdd9-ffb9bd38b10b" />
<img width="1311" height="663" alt="image" src="https://github.com/user-attachments/assets/62ccbf4c-6ff4-4f63-8a78-3128475c3d3e" />

---

## Submission

- **Repository:** <!-- ADD GITHUB REPO LINK --> https://github.com/Varad1722/Mobile-Robotics_Assignments
- **Commit hash:** <!-- ADD COMMIT HASH --> 990257a


