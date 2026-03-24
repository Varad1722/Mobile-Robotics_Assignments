"""
Cylinder Detection Pipeline — ROS 2 Node
Assignment 1: Semantic Landmark Extraction and Classification

Pipeline stages:
  0. Box filter + voxel downsample            (preprocessing)
  1. Normal estimation (SVD) + RANSAC plane removal
  2. Euclidean clustering (BFS + cKDTree)
  3. Cylinder RANSAC per cluster
  4. HSV color classification (Red / Green / Blue)
"""

import collections
import numpy as np
from scipy.spatial import cKDTree

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


# ==========================================
# CONFIGURATION CLASS
# ==========================================

class PipelineConfig:
    """
    Holds all tunable parameters for the point cloud processing pipeline.
    Adjust these values to tune detection performance per bag file.
    """
    def __init__(self):
        # Topic settings
        self.topic = '/oakd/points'

        # Passthrough / Box Filter — discard points outside this XYZ workspace
        self.box_min = np.array([-1.0, -0.6,  0.2])
        self.box_max = np.array([ 1.0,  0.6,  2.0])

        # Voxel downsampling — 3 cm grid cells
        # Tested 0.01 (too dense, slow), 0.02 (good), 0.03 (optimal speed/detail balance)
        self.voxel_size = 0.03

        # Plane RANSAC — floor / ceiling removal
        self.floor_dist         = 0.04                        # inlier threshold (m) — increased from 0.02 for more aggressive floor removal
        self.target_normal      = np.array([0.0, 1.0, 0.0])  # expected floor normal (Y-up)
        self.normal_thresh      = 0.85                        # min |dot| with vertical
        self.num_plane_removals = 3                           # planes stripped per frame

        # Euclidean clustering
        self.cluster_radius  = 0.06    # BFS neighbor search radius (m)
        self.cluster_min_pts = 20      # lowered from 50 to handle sparse clusters in complex scenes
        self.cluster_max_pts = 5000    # maximum points per cluster

        # Cylinder RANSAC
        self.cyl_radius      = 0.055   # expected cylinder radius (m)
        self.cyl_inlier_tol  = 0.030   # tolerance around expected radius — widened from 0.015 for noisy/sparse clusters
        self.cyl_axis_thresh = 0.70    # min |dot| of axis with vertical — relaxed from 0.80 to handle slight camera tilt
        self.cyl_min_inliers = 10      # minimum inliers to accept a detection — lowered from 20 for sparse clusters
        self.max_cylinders   = 3       # maximum cylinders detected per frame

        # RViz display settings
        # Point size for PointCloud2 displays — set to 0.03 in RViz (Style: Flat Squares)
        # Background color set to dark grey (48, 48, 48) in RViz Global Options
        self.marker_alpha    = 0.9     # cylinder marker opacity — increased from 0.8 for better visibility


# ==========================================
# VISUALIZER CLASS
# ==========================================

class CylinderVisualizer:
    """
    Handles creation and publishing of RViz MarkerArrays
    that represent the detected cylinders.
    """
    def __init__(self, publisher, cfg: PipelineConfig):
        self.pub_markers = publisher
        self.cfg         = cfg

    def create_cylinder_marker(self, center, radius, rgb, marker_id, frame_id):
        m = Marker()
        m.header.frame_id = frame_id
        m.id               = marker_id
        m.type             = Marker.CYLINDER
        m.action           = Marker.ADD

        m.pose.position.x = float(center[0])
        m.pose.position.y = float(0.0)   # snap to floor level for visualization
        m.pose.position.z = float(center[2])

        # Identity quaternion — upright orientation in oakd_rgb_camera_optical_frame
        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 1.0

        m.scale.x = float(radius * 2.0)
        m.scale.y = float(radius * 2.0)
        m.scale.z = 0.4

        m.color.r = float(rgb[0])
        m.color.g = float(rgb[1])
        m.color.b = float(rgb[2])
        m.color.a = float(self.cfg.marker_alpha)   # configurable opacity
        return m

    def publish_viz(self, cylinders, frame_id):
        ma = MarkerArray()

        # Clear all previous markers before publishing new ones
        clear_marker        = Marker()
        clear_marker.action = Marker.DELETEALL
        ma.markers.append(clear_marker)

        for i, (model, rgb, name) in enumerate(cylinders):
            center, _, radius = model
            marker = self.create_cylinder_marker(
                center, radius, rgb, 2000 + i, frame_id)
            ma.markers.append(marker)

        self.pub_markers.publish(ma)


# ==========================================
# PIPELINE LOGIC
# ==========================================

class CylinderPipeline:
    """
    All geometric processing stages.
    No ROS dependencies — pure NumPy and cKDTree only.
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Color helpers
    # ------------------------------------------------------------------

    def rgb_to_hsv(self, r, g, b):
        """
        Converts a single RGB point to HSV color space.

        :param r: Red component   (0.0 - 1.0)
        :param g: Green component (0.0 - 1.0)
        :param b: Blue component  (0.0 - 1.0)
        :return: Tuple (h, s, v) where H is [0, 360], S and V are [0, 1]
        """
        mx = max(r, g, b)
        mn = min(r, g, b)
        df = mx - mn

        # Calculate Hue
        if mx == mn:
            h = 0.0
        elif mx == r:
            h = (60.0 * ((g - b) / df) + 360.0) % 360.0
        elif mx == g:
            h = (60.0 * ((b - r) / df) + 120.0) % 360.0
        else:
            h = (60.0 * ((r - g) / df) + 240.0) % 360.0

        # Calculate Saturation
        s = 0.0 if mx == 0.0 else (df / mx)

        # Calculate Value
        v = mx

        return h, s, v

    def classify_color(self, h, s, v):
        """
        Map HSV values to a semantic color label.

        Thresholds derived from actual terminal HSV readings on rgbd_bag_2:

          Red   : h < 15° or h > 320°,   s >= 0.10
                  Wraps around 0/360 on the hue circle.
                  (measured: h=335–342, s=0.17–0.23, v=0.65–0.82)

          Green : 90° <= h <= 150°,       s >= 0.10
                  (measured: h=91–100, s=0.17–0.25, v=0.53–0.63)

          Blue  : 200° <= h <= 270°,      s >= 0.10
                  (measured: h=210–221, s=0.48–0.70, v=0.39–0.44)

        A saturation guard (s < 0.10) and value guard (v < 0.15) prevent
        misclassifying grey noise or dark unlit surfaces.
        """
        # Reject pure black / unlit surfaces
        if s < 0.10 or v < 0.15:
            return "unknown", [0.5, 0.5, 0.5]

        if h < 15.0 or h > 320.0:
            return "red",   [1.0, 0.0, 0.0]
        elif 90.0 <= h <= 150.0:
            return "green", [0.0, 1.0, 0.0]
        elif 200.0 <= h <= 270.0:
            return "blue",  [0.0, 0.0, 1.0]
        else:
            return "unknown", [0.5, 0.5, 0.5]

    # ------------------------------------------------------------------
    # Neighbor search
    # ------------------------------------------------------------------

    def get_neighbors(self, pts, queries, k=15):
        """
        Calculates k-nearest neighbors using a KDTree.

        :param pts:     The source point cloud (Nx3).
        :param queries: The points for which we want neighbors (Mx3).
        :param k:       Number of neighbors to find.
        :return:        Indices of neighbors in the 'pts' array, shape (M, k).
        """
        if len(pts) < k:
            return None
        tree = cKDTree(pts)
        _, idxs = tree.query(queries, k=k)
        return idxs

    # ------------------------------------------------------------------
    # Task 0a — Box filter
    # ------------------------------------------------------------------

    def box_filter(self, pts, colors):
        """
        Removes points outside the specified XYZ bounding box.

        Single boolean mask — O(N), no Python loops.

        :param pts:    Input XYZ array (N x 3).
        :param colors: Input RGB array (N x 3).
        :return:       Tuple of (filtered_pts, filtered_colors).
        """
        cfg  = self.cfg
        mask = (
            (pts[:, 0] >= cfg.box_min[0]) & (pts[:, 0] <= cfg.box_max[0]) &
            (pts[:, 1] >= cfg.box_min[1]) & (pts[:, 1] <= cfg.box_max[1]) &
            (pts[:, 2] >= cfg.box_min[2]) & (pts[:, 2] <= cfg.box_max[2])
        )
        return pts[mask], colors[mask]

    # ------------------------------------------------------------------
    # Task 0b — Voxel downsample
    # ------------------------------------------------------------------

    def downsample(self, pts, colors):
        """
        Reduces point cloud density using a voxel grid approach.

        Implementation: Convert points to integer voxel coordinates by
        dividing by voxel_size and flooring. np.unique with return_index=True
        keeps the first point found in each unique voxel.

        :param pts:    Input XYZ array.
        :param colors: Input RGB array.
        :return:       Tuple (downsampled_pts, downsampled_colors).
        """
        voxel_coords = np.floor(pts / self.cfg.voxel_size).astype(np.int32)
        _, first_idx = np.unique(voxel_coords, axis=0, return_index=True)
        return pts[first_idx], colors[first_idx]

    # ------------------------------------------------------------------
    # Task 1a — Normal estimation
    # ------------------------------------------------------------------

    def estimate_normals(self, pts, k=15):
        """
        Estimates a surface normal for every point using SVD on the
        k-nearest-neighbor patch.

        Implementation:
          1. For each point, find k neighbors with cKDTree.
          2. Center the neighborhood by subtracting its mean.
          3. Apply SVD (np.linalg.svd) on the centered (k x 3) matrix.
          4. The last row of Vt is the direction of minimum variance
             which is perpendicular to the local surface — the normal.

        :param pts: Input XYZ array (N x 3).
        :param k:   Neighborhood size.
        :return:    Normal array (N x 3).
        """
        n_pts   = len(pts)
        normals = np.zeros((n_pts, 3), dtype=np.float64)

        neighbor_idx = self.get_neighbors(pts, pts, k=k)
        if neighbor_idx is None:
            return normals

        for i in range(n_pts):
            patch    = pts[neighbor_idx[i]]
            centered = patch - patch.mean(axis=0)
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            normals[i] = Vt[-1]   # eigenvector of smallest eigenvalue = normal

        return normals

    # ------------------------------------------------------------------
    # Task 1b — Plane RANSAC
    # ------------------------------------------------------------------

    def find_plane_ransac(self, pts, iters=100):
        """
        Fits a plane model (ax + by + cz + d = 0) to the cloud using RANSAC.

        Implementation:
          1. Sample 3 random points to define a candidate plane.
          2. Compute normal via cross product and normalise.
          3. Check alignment with self.cfg.target_normal — reject walls.
          4. Count inliers: points within self.cfg.floor_dist of the plane.
          5. Return the model with the most inliers.

        :return: Tuple (normal, d, inlier_mask), or (None, None, None).
        """
        cfg   = self.cfg
        n_pts = len(pts)

        best_count  = 0
        best_normal = None
        best_d      = None
        best_mask   = None

        for _ in range(iters):
            idx        = np.random.choice(n_pts, 3, replace=False)
            p1, p2, p3 = pts[idx[0]], pts[idx[1]], pts[idx[2]]

            normal     = np.cross(p2 - p1, p3 - p1)
            normal_len = np.linalg.norm(normal)
            if normal_len < 1e-6:
                continue
            normal = normal / normal_len

            # Reject planes that are not roughly horizontal
            if abs(np.dot(normal, cfg.target_normal)) < cfg.normal_thresh:
                continue

            d     = -np.dot(normal, p1)
            dists = np.abs(pts @ normal + d)
            mask  = dists < cfg.floor_dist
            count = int(np.sum(mask))

            if count > best_count:
                best_count  = count
                best_normal = normal
                best_d      = d
                best_mask   = mask

        return best_normal, best_d, best_mask

    # ------------------------------------------------------------------
    # Task 2 — Euclidean clustering
    # ------------------------------------------------------------------

    def euclidean_clustering(self, pts):
        """
        Groups points into distinct objects using BFS and cKDTree radius search.

        Every unvisited point seeds a new cluster. All neighbors within
        cluster_radius are added to the BFS queue and marked visited.
        Clusters outside the [min, max] point count range are discarded.

        :param pts: Input XYZ array (N x 3).
        :return:    List of integer index arrays, one per valid cluster.
        """
        cfg     = self.cfg
        n_pts   = len(pts)
        visited = np.zeros(n_pts, dtype=bool)
        tree    = cKDTree(pts)
        clusters = []

        for seed in range(n_pts):
            if visited[seed]:
                continue

            cluster_indices = []
            queue = collections.deque([seed])
            visited[seed] = True

            while queue:
                current = queue.popleft()
                cluster_indices.append(current)
                neighbors = tree.query_ball_point(pts[current], r=cfg.cluster_radius)
                for nb in neighbors:
                    if not visited[nb]:
                        visited[nb] = True
                        queue.append(nb)

            size = len(cluster_indices)
            if cfg.cluster_min_pts <= size <= cfg.cluster_max_pts:
                clusters.append(np.array(cluster_indices, dtype=np.int32))

        return clusters

    # ------------------------------------------------------------------
    # Task 3 — Cylinder RANSAC
    # ------------------------------------------------------------------

    def find_single_cylinder(self, pts, normals, iters=500):
        """
        Fits a cylinder model to a cluster using RANSAC.

        On a cylinder surface, normals point radially outward — perpendicular
        to the axis. The cross product of two such normals gives the axis.

        Implementation:
          1. Sample 2 points and their surface normals.
          2. axis = cross(n1, n2) — verify vertical alignment with cyl_axis_thresh.
          3. Project all points onto the axis; compute perpendicular distances.
          4. Inliers: |perp_dist - cyl_radius| < cyl_inlier_tol.

        All distance calculations use np.cross and np.linalg.norm — fully
        vectorised, no inner loops.
        Iterations increased to 500 to improve detection on sparse clusters
        like the pink cylinder which has fewer surface points.

        :return: (center, axis, radius, inlier_mask) or None if not found.
        """
        cfg      = self.cfg
        n_pts    = len(pts)
        vertical = cfg.target_normal

        if n_pts < 10:   # lowered from 20 to handle sparse pink cluster
            return None

        best_count  = 0
        best_result = None

        for _ in range(iters):
            idx    = np.random.choice(n_pts, 2, replace=False)
            p1, p2 = pts[idx[0]], pts[idx[1]]
            n1, n2 = normals[idx[0]], normals[idx[1]]

            axis     = np.cross(n1, n2)
            axis_len = np.linalg.norm(axis)
            if axis_len < 1e-6:
                continue
            axis = axis / axis_len

            # Ensure axis points upward consistently
            if np.dot(axis, vertical) < 0:
                axis = -axis

            # Reject axes that are not approximately vertical
            if abs(np.dot(axis, vertical)) < cfg.cyl_axis_thresh:
                continue

            # Perpendicular distance from every point to the axis through p1
            V          = pts - p1
            proj_len   = V @ axis
            perp_vecs  = V - np.outer(proj_len, axis)
            perp_dists = np.linalg.norm(perp_vecs, axis=1)

            inlier_mask = np.abs(perp_dists - cfg.cyl_radius) < cfg.cyl_inlier_tol
            count       = int(np.sum(inlier_mask))

            if count > best_count:
                best_count    = count
                inlier_pts    = pts[inlier_mask]
                center        = inlier_pts.mean(axis=0)
                actual_radius = float(perp_dists[inlier_mask].mean())
                best_result   = (center, axis, actual_radius, inlier_mask)

        if best_count < cfg.cyl_min_inliers:
            return None

        return best_result


# ==========================================
# ROS NODE
# ==========================================

class CylinderProcessorNode(Node):

    def __init__(self):
        super().__init__('cylinder_processor_node')
        self.cfg      = PipelineConfig()
        self.pipeline = CylinderPipeline(self.cfg)

        # Debug publishers — subscribe in RViz to inspect each pipeline stage
        self.pub_stage0 = self.create_publisher(
            PointCloud2, 'pipeline/stage0_box', 10)
        self.pub_stage1 = self.create_publisher(
            PointCloud2, 'pipeline/stage1_no_planes', 10)
        self.pub_stage3 = self.create_publisher(
            PointCloud2, 'pipeline/stage3_candidates', 10)

        # Final detection result publisher
        marker_pub      = self.create_publisher(MarkerArray, 'viz/detections', 10)
        self.visualizer = CylinderVisualizer(marker_pub, self.cfg)

        self.sub = self.create_subscription(
            PointCloud2, self.cfg.topic, self.listener_callback, 10)

        self.get_logger().info("CylinderProcessorNode ready — waiting for point clouds.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _decode_colors(self, packed_col):
        """
        Unpack float32-encoded RGB from the OAK-D point cloud.
        Three 8-bit channels are stored as bytes inside a uint32
        cast to float32 at byte offset 16 (column index 4).
        """
        rgb_u32 = packed_col.view(np.uint32)
        r = ((rgb_u32 >> 16) & 0xFF).astype(np.float32) / 255.0
        g = ((rgb_u32 >>  8) & 0xFF).astype(np.float32) / 255.0
        b = ( rgb_u32        & 0xFF).astype(np.float32) / 255.0
        return np.stack([r, g, b], axis=1)

    def numpy_to_pc2_rgb(self, pts, colors, frame_id):
        """
        Converts Nx3 XYZ coordinates and Nx3 RGB color arrays into a
        ROS 2 PointCloud2 message.

        This utility handles the conversion of floating-point spatial data
        and the packing of three 8-bit color channels (R, G, B) into a
        single 32-bit float field, which is the standard format for RGB
        point clouds in ROS and RViz.

        :param pts:      A numpy array of shape (N, 3) containing [x, y, z].
        :param colors:   A numpy array of shape (N, 3) containing [r, g, b] (0.0 to 1.0).
        :param frame_id: The TF frame string for the message header.
        :return:         A sensor_msgs/PointCloud2 message ready for publishing.
        """
        msg             = PointCloud2()
        msg.header.frame_id = frame_id
        msg.height      = 1
        msg.width       = len(pts)
        msg.fields      = [
            PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step   = 16
        msg.row_step     = 16 * len(pts)
        msg.is_dense     = True

        c          = (np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint32)
        rgb_packed = ((c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]).view(np.float32)
        data       = np.hstack([pts.astype(np.float32), rgb_packed.reshape(-1, 1)])
        msg.data   = data.tobytes()
        return msg

    def _publish_colored_clusters(self, clusters, work_pts, frame_id):
        """Assign a distinct debug color to each cluster and publish for RViz."""
        palette = np.array([
            [1.0, 0.2, 0.2], [0.2, 1.0, 0.2], [0.2, 0.2, 1.0],
            [1.0, 1.0, 0.2], [1.0, 0.2, 1.0], [0.2, 1.0, 1.0],
        ])
        all_pts, all_colors = [], []
        for i, idx_arr in enumerate(clusters):
            color = palette[i % len(palette)]
            all_pts.append(work_pts[idx_arr])
            all_colors.append(np.tile(color, (len(idx_arr), 1)))
        if all_pts:
            self.pub_stage3.publish(self.numpy_to_pc2_rgb(
                np.vstack(all_pts), np.vstack(all_colors), frame_id))

    # ------------------------------------------------------------------
    # Main callback
    # ------------------------------------------------------------------

    def listener_callback(self, msg):
        """
        Main ROS Callback. Orchestrates the flow from PointCloud2
        to cylinder detections with semantic color labels.
        """
        frame_id = msg.header.frame_id

        # Parse raw PointCloud2 binary data
        stride   = msg.point_step // 4
        raw_data = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, stride)
        pts      = raw_data[:, :3].copy()

        # Remove NaN / Inf points
        finite_mask = np.all(np.isfinite(pts), axis=1)
        pts         = pts[finite_mask]

        # Decode packed RGB (byte offset 16 = float32 column 4)
        raw_colors = self._decode_colors(raw_data[finite_mask, 4].copy())

        if len(pts) < 100:
            return

        # ----- Stage 0a: Box filter -----
        pts_box, colors_box = self.pipeline.box_filter(pts, raw_colors)
        if len(pts_box) < 100:
            self.get_logger().warn(
                f"Box filter left only {len(pts_box)} points — check box_min/box_max.")
            return

        # ----- Stage 0b: Voxel downsample -----
        pts_v, colors_v = self.pipeline.downsample(pts_box, colors_box)
        if len(pts_v) < 50:
            self.get_logger().warn("Downsample produced fewer than 50 points.")
            return

        # Publish post-downsample cloud for RViz debugging
        self.pub_stage0.publish(
            self.numpy_to_pc2_rgb(pts_v, colors_v, frame_id))

        # ----- Stage 1a: Normal estimation -----
        normals = self.pipeline.estimate_normals(pts_v, k=15)

        # ----- Stage 1b: Iterative plane removal (floor, table, ceiling) -----
        work_pts     = pts_v.copy()
        work_colors  = colors_v.copy()
        work_normals = normals.copy()

        for _ in range(self.cfg.num_plane_removals):
            if len(work_pts) < 50:
                break
            _, _, inlier_mask = self.pipeline.find_plane_ransac(work_pts)
            if inlier_mask is None or int(np.sum(inlier_mask)) < 20:
                break
            keep         = ~inlier_mask
            work_pts     = work_pts[keep]
            work_colors  = work_colors[keep]
            work_normals = work_normals[keep]

        # Publish plane-removed cloud for RViz debugging
        self.pub_stage1.publish(
            self.numpy_to_pc2_rgb(work_pts, work_colors, frame_id))

        if len(work_pts) < 20:
            self.visualizer.publish_viz([], frame_id)
            return

        # ----- Stage 2: Euclidean clustering -----
        clusters = self.pipeline.euclidean_clustering(work_pts)
        if not clusters:
            self.visualizer.publish_viz([], frame_id)
            return

        # Publish cluster-colored cloud for RViz debugging
        self._publish_colored_clusters(clusters, work_pts, frame_id)

        # ----- Stages 3 + 4: Cylinder RANSAC + color classification -----
        # Final detections format: list of ((center, axis, radius), rgb_color, name)
        detected_cylinders = []

        for cluster_idx_array in clusters:
            if len(detected_cylinders) >= self.cfg.max_cylinders:
                break

            cluster_pts     = work_pts[cluster_idx_array]
            cluster_colors  = work_colors[cluster_idx_array]
            cluster_normals = work_normals[cluster_idx_array]

            # Debug: log cluster size and center for tuning and verification
            self.get_logger().info(
                f"Cluster: {len(cluster_pts)} pts  "
                f"center=({cluster_pts.mean(axis=0)[0]:.2f}, "
                f"{cluster_pts.mean(axis=0)[1]:.2f}, "
                f"{cluster_pts.mean(axis=0)[2]:.2f})"
            )

            result = self.pipeline.find_single_cylinder(cluster_pts, cluster_normals)
            if result is None:
                continue

            center, axis, radius, inlier_mask = result

            # Use only the cylinder surface inlier points for color classification
            avg_rgb = cluster_colors[inlier_mask].mean(axis=0)
            h, s, v = self.pipeline.rgb_to_hsv(
                float(avg_rgb[0]), float(avg_rgb[1]), float(avg_rgb[2]))
            label, display_rgb = self.pipeline.classify_color(h, s, v)

            # HSV debug log — printed per detection to verify color classification
            self.get_logger().info(
                f"  HSV debug: h={h:.1f}  s={s:.3f}  v={v:.3f}  → label={label}"
            )

            model = (center, axis, radius)
            detected_cylinders.append((model, display_rgb, label))

            self.get_logger().info(
                f"Cylinder detected: label={label}  "
                f"center=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})  "
                f"radius={radius:.3f} m  inliers={int(np.sum(inlier_mask))}"
            )

        # --- Deduplication: remove cylinders too close to each other ---
        # If two detections are within 0.20 m of each other on the XZ plane,
        # keep the first (largest cluster wins — earlier clusters have more points).
        # This prevents one physical cylinder generating two overlapping markers
        # when sparse point coverage causes it to split into nearby clusters.
        MIN_SEPARATION = 0.20   # meters
        deduped = []
        for cyl in detected_cylinders:
            c_center = cyl[0][0]
            too_close = False
            for kept in deduped:
                k_center = kept[0][0]
                dist = float(np.linalg.norm(
                    np.array([c_center[0], c_center[2]])
                    - np.array([k_center[0], k_center[2]])
                ))
                if dist < MIN_SEPARATION:
                    too_close = True
                    self.get_logger().warn(
                        f"Duplicate suppressed: {cyl[2]} too close to {kept[2]} "
                        f"(dist={dist:.3f} m)"
                    )
                    break
            if not too_close:
                deduped.append(cyl)

        self.visualizer.publish_viz(deduped, frame_id)


# ==========================================
# ENTRY POINT
# ==========================================

def main():
    rclpy.init()
    node = CylinderProcessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
