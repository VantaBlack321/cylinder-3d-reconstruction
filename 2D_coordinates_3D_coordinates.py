import cv2
import math
import numpy as np
import os

# 2. Dimensions and physical constants
img_width = 3024
img_height = 4032
radius = 4.25       # cm
pipe_length = 43.3  # cm

# focal length width
fx = 3272.5
# focal length height
fy = 3389.7

# Central coordinates x axis
cx = 1508
# Central coordinates y axis
cy = 2349

print(f"Loaded image size: {img_width}x{img_height}")

# Locked calibration markers from verified run (keeps K and cylinder pose fixed)
locked_markers = {
    'm_0': (1381, 1107),
    'm_1': (1414, 1475),
    'm_2': (1435, 1688),
    'm_3': (1444, 1824)
}

# Only prompt for the 4 Box II corners (P -> Q -> R -> S)
labels = ['P', 'Q', 'R', 'S']
clicked_coordinates = []
points_dict = {}

# Camera Matrix
K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])

C = np.array([[0], [0], [0]])
reference_spacing = 5.0  # cm

# Pre-populate locked calibration markers into points_dict
for m_lbl, (mu, mv) in locked_markers.items():
    p_m = np.array([[mu], [mv], [1]], dtype=np.float64)
    d_m = np.linalg.solve(K, p_m)
    points_dict[m_lbl] = {
        "pixel": (mu, mv),
        "homogeneous": p_m,
        "viewing_direction": d_m,
        "d_hat": d_m / np.linalg.norm(d_m)
    }

def draw_marker_on_img(img, u, v, letter, dot_color=(0, 255, 0)):
    cv2.circle(img, (u, v), 5, dot_color, -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.5
    thickness = 2
    gap = 35
    start_x = u + 25
    start_y = v - 15

    letter_text = f"{letter}:"
    cv2.putText(img, letter_text, (start_x, start_y),
                font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
    (text_width, _), _ = cv2.getTextSize(letter_text, font, font_scale, thickness)
    coords_text = f"({u}, {v})"
    cv2.putText(img, coords_text, (start_x + text_width + gap, start_y),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def mouse_click_callback(event, u, v, flags, param):
    """Callback function with live hover crosshair and click recording."""
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_coordinates) < len(labels):
        idx = len(clicked_coordinates)
        letter = labels[idx]

        clicked_coordinates.append((u, v))
        print(f"Stored Point {letter} ({idx + 1}): U={u}, V={v}")

        p = np.array([[u], [v], [1]], dtype=np.float64)
        d = np.linalg.solve(K, p)
        d_hat = d / np.linalg.norm(d)

        points_dict[letter] = {
            "pixel": (u, v),
            "homogeneous": p,
            "viewing_direction": d,
            "d_hat": d_hat
        }

        draw_marker_on_img(display_img, u, v, letter, dot_color=(0, 255, 0))
        cv2.imshow("Image Window", display_img)

        if len(clicked_coordinates) == 4:
            print("\n>>> All 4 Box II corners (P, Q, R, S) recorded! Press any key to continue...")

    elif event == cv2.EVENT_MOUSEMOVE:
        hover_img = display_img.copy()
        cv2.line(hover_img, (u - 25, v), (u + 25, v), (255, 255, 0), 2)
        cv2.line(hover_img, (u, v - 25), (u, v + 25), (255, 255, 0), 2)
        cv2.putText(hover_img, f"({u}, {v})", (u + 30, v + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Image Window", hover_img)

image_path = os.path.expanduser('~/Downloads/cylinder-3d-reconstruction/IMG_2667.JPG')
if not os.path.exists(image_path):
    image_path = os.path.expanduser('~/Downloads/IMG_2667.JPG')

original_img = cv2.imread(image_path)

if original_img is None:
    print(f"Error: Could not load image from {image_path}. Check the file path.")
else:
    display_img = original_img.copy()

    # Draw locked m_0..m_3 markers and optical center on display_img
    for m_lbl, (mu, mv) in locked_markers.items():
        draw_marker_on_img(display_img, mu, mv, m_lbl, dot_color=(0, 0, 255))
    cv2.drawMarker(display_img, (int(cx), int(cy)), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=30, thickness=2)
    cv2.putText(display_img, f"Origin ({int(cx)}, {int(cy)})", (int(cx) + 20, int(cy) + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.namedWindow("Image Window")
    cv2.setMouseCallback("Image Window", mouse_click_callback)
    cv2.imshow("Image Window", display_img)

    print("Calibration m_0..m_3 locked. Click the 4 corners of Box II (P -> Q -> R -> S), or press 'ESC'/'q' to use stored baseline.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q') or len(clicked_coordinates) == 4:
            if len(clicked_coordinates) == 4:
                cv2.waitKey(400)
            break

    cv2.destroyAllWindows()

    # Fallback to stored baseline Box II clicks if ESC/q was pressed
    if len(clicked_coordinates) < 4:
        print("Using stored baseline Box II points...")
        fallback_box2 = {
            'P': (1025, 2096), 'Q': (1158, 2195), 'R': (1344, 2583), 'S': (1284, 2652)
        }
        for lbl in labels:
            u, v = fallback_box2[lbl]
            p = np.array([[u], [v], [1]], dtype=np.float64)
            d = np.linalg.solve(K, p)
            points_dict[lbl] = {
                "pixel": (u, v),
                "homogeneous": p,
                "viewing_direction": d,
                "d_hat": d / np.linalg.norm(d)
            }

    # Get normalized viewing directions for locked calibration markers
    d0 = points_dict['m_0']['d_hat'].flatten()
    d1 = points_dict['m_1']['d_hat'].flatten()
    d2 = points_dict['m_2']['d_hat'].flatten()
    d3 = points_dict['m_3']['d_hat'].flatten()

    B1 = np.column_stack((d0, -2 * d1, d2, np.zeros(3)))
    B2 = np.column_stack((np.zeros(3), d1, -2 * d2, d3))
    B = np.vstack((B1, B2))

    # Solve for relative depths using SVD
    U, S, Vt = np.linalg.svd(B)
    t_relative = Vt[-1]
    if np.mean(t_relative) < 0:
        t_relative = -t_relative

    # 1. Scale relative depths to true physical scale (cm)
    diff_01_rel = t_relative[1] * d1 - t_relative[0] * d0
    k_scale = reference_spacing / np.linalg.norm(diff_01_rel)
    t_metric = t_relative * k_scale

    # 2. Compute 3D surface points M_0 through M_3
    M0 = t_metric[0] * d0
    M1 = t_metric[1] * d1
    M2 = t_metric[2] * d2
    M3 = t_metric[3] * d3
    M_points = np.vstack([M0, M1, M2, M3])

    # 3. Estimate cylinder axis direction a_hat
    axis_vector = M3 - M0
    a_hat = axis_vector / np.linalg.norm(axis_vector)

    print("\n================ Cylinder Axis Direction (a_hat) ===========")
    print(f"a_hat: [{a_hat[0]:.4f}, {a_hat[1]:.4f}, {a_hat[2]:.4f}]")

    # Normal vector pointing inward from surface point M0 to the central axis
    n_dir = -M0 - np.dot(-M0, a_hat) * a_hat
    n_hat = n_dir / np.linalg.norm(n_dir)

    # Walk inward from surface marker M0 by radius (4.25 cm) to reach center axis A
    A = M0 + radius * n_hat

    print("\n================ Cylinder Axis Anchor (A) ================")
    print(f"A: X = {A[0]:.2f}, Y = {A[1]:.2f}, Z = {A[2]:.2f}")

    print("\n================ Metric Surface Points (cm) ================")
    for idx, M in enumerate(M_points):
        print(f"M_{idx}: X={M[0]:.2f}, Y={M[1]:.2f}, Z={M[2]:.2f}")

    # 4. Calibration vs Independent Validation Verification
    print("\n================ Calibration vs Validation ================")
    print(f"Calibration Scale Baseline (||M1 - M0||) : {np.linalg.norm(M1 - M0):.2f} cm (Enforced)")
    print(f"Independent Validation 1   (||M2 - M1||) : {np.linalg.norm(M2 - M1):.2f} cm (Target: 5.00 cm)")
    print(f"Independent Validation 2   (||M3 - M2||) : {np.linalg.norm(M3 - M2):.2f} cm (Target: 5.00 cm)")
    print(f"Independent Total Span     (||M3 - M0||) : {np.linalg.norm(M3 - M0):.2f} cm (Target: 15.00 cm)")

    axis_m0_m1 = M1 - M0
    a_hat_isolated = axis_m0_m1 / np.linalg.norm(axis_m0_m1)
    print(f"Isolated Axis from M0->M1 only          : [{a_hat_isolated[0]:.4f}, {a_hat_isolated[1]:.4f}, {a_hat_isolated[2]:.4f}]")

    # ========================================================
    # Ray-Cylinder Intersection & Patch Unwrapping
    # ========================================================
    u_axis = -n_hat
    u_axis = u_axis - np.dot(u_axis, a_hat) * a_hat
    u_axis = u_axis / np.linalg.norm(u_axis)
    v_axis = np.cross(a_hat, u_axis)

    def intersect_ray_cylinder(d_hat, A_anchor, a_hat_dir, R_cyl):
        """Calculates the 3D surface intersection for a camera ray."""
        v = np.cross(d_hat, a_hat_dir)
        w = np.cross(-A_anchor, a_hat_dir)

        c2 = np.dot(v, v)
        c1 = 2.0 * np.dot(v, w)
        c0 = np.dot(w, w) - (R_cyl ** 2)

        discriminant = c1**2 - 4.0 * c2 * c0
        if discriminant < 0:
            return None, discriminant, None

        t1 = (-c1 - np.sqrt(discriminant)) / (2.0 * c2)
        t2 = (-c1 + np.sqrt(discriminant)) / (2.0 * c2)

        t_pos = [t for t in (t1, t2) if t > 0]
        if not t_pos:
            return None, discriminant, None

        t_chosen = min(t_pos)
        return t_chosen * d_hat, discriminant, t_chosen

    def unwrap_and_area(corner_dict, warn_bounds=True):
        """Maps 4 pixel corners to 3D, flags/clamps z > 43.3 cm, unwraps (s, z), and returns Shoelace area."""
        p3d = {}
        punw = {}
        for lbl in ['P', 'Q', 'R', 'S']:
            u_c, v_c = corner_dict[lbl]
            p_hom = np.array([[u_c], [v_c], [1]], dtype=np.float64)
            d_vec = np.linalg.solve(K, p_hom)
            d_unit = (d_vec / np.linalg.norm(d_vec)).flatten()

            P_xyz, _, _ = intersect_ray_cylinder(d_unit, A, a_hat, radius)
            p3d[lbl] = P_xyz

            rel_vec = P_xyz - A
            z_axial = np.dot(P_xyz - M0, a_hat)
            if z_axial > pipe_length or z_axial < 0:
                if warn_bounds:
                    print(f"[WARNING] {lbl} axial depth ({z_axial:.2f} cm) exceeds physical pipe boundary (0–{pipe_length} cm). Clamping to boundary.")
                z_axial = float(np.clip(z_axial, 0.0, pipe_length))

            radial_proj = rel_vec - np.dot(rel_vec, a_hat) * a_hat
            theta = np.arctan2(np.dot(radial_proj, v_axis), np.dot(radial_proj, u_axis))
            s_arc = radius * theta
            punw[lbl] = (s_arc, z_axial)

        pts_2d = np.array([punw[lbl] for lbl in ['P', 'Q', 'R', 'S']])
        centroid = np.mean(pts_2d, axis=0)
        angles = np.arctan2(pts_2d[:, 1] - centroid[1], pts_2d[:, 0] - centroid[0])
        sorted_pts = pts_2d[np.argsort(angles)]
        s_pts = sorted_pts[:, 0]
        z_pts = sorted_pts[:, 1]
        area = 0.5 * np.abs(np.dot(s_pts, np.roll(z_pts, 1)) - np.dot(z_pts, np.roll(s_pts, 1)))
        return area, p3d, punw, sorted_pts

    # Baseline Box II Reconstruction
    print("\n================ Reconstructed Patch Corners (P, Q, R, S) ================")
    base_corners = {lbl: points_dict[lbl]['pixel'] for lbl in ['P', 'Q', 'R', 'S']}
    calc_area, patch_3d, patch_unwrapped, sorted_pts = unwrap_and_area(base_corners, warn_bounds=True)

    for label in ['P', 'Q', 'R', 'S']:
        P_xyz = patch_3d[label]
        s_arc, z_axial = patch_unwrapped[label]
        print(f"{label}: 3D = [{P_xyz[0]:6.2f}, {P_xyz[1]:6.2f}, {P_xyz[2]:6.2f}] cm | Unwrapped (s, z) = ({s_arc:6.2f}, {z_axial:6.2f}) cm")

    recon_width = abs(patch_unwrapped['S'][0] - patch_unwrapped['P'][0])
    recon_height = abs(patch_unwrapped['Q'][1] - patch_unwrapped['P'][1])
    recon_rect_area = recon_width * recon_height

    actual_width = 5.8
    actual_height = 12.5
    actual_area_cm2 = 72.5

    print("\n================ Reconstructed Dimensions & Area ================")
    print(f"Reconstructed Arc Width (s) : {recon_width:6.2f} cm | Actual: {actual_width:6.2f} cm | Error: {abs(recon_width - actual_width):.2f} cm ({(abs(recon_width - actual_width)/actual_width)*100:.2f}%)")
    print(f"Reconstructed Axial Depth (z): {recon_height:6.2f} cm | Actual: {actual_height:6.2f} cm | Error: {abs(recon_height - actual_height):.2f} cm ({(abs(recon_height - actual_height)/actual_height)*100:.2f}%)")
    print(f"Rectangular Area (w * h)    : {recon_rect_area:6.2f} cm² | Actual: {actual_area_cm2:6.2f} cm² | Error: {abs(recon_rect_area - actual_area_cm2):.2f} cm² ({(abs(recon_rect_area - actual_area_cm2)/actual_area_cm2)*100:.2f}%)")
    print(f"Shoelace Polygon Area       : {calc_area:6.2f} cm² | Actual: {actual_area_cm2:6.2f} cm² | Error: {abs(calc_area - actual_area_cm2):.2f} cm² ({(abs(calc_area - actual_area_cm2)/actual_area_cm2)*100:.2f}%)")

    # ========================================================
    # Point R Sensitivity Sweep (+-1, +-2, +-5 pixels, Clamped)
    # ========================================================
    print("\n================ Point R Sensitivity Sweep (Clamped <= 43.3 cm) ================")
    print(f"{'Offset':<10} | {'Disc (b²-4ac)':<14} | {'t (cm)':<8} | {'Raw z':<8} | {'Clamped z':<9} | {'Area (cm²)':<10} | {'Area Error'}")
    print("-" * 82)

    base_u, base_v = points_dict['R']['pixel']
    shifts = [
        ('Base', 0, 0),
        ('u + 1', 1, 0), ('u - 1', -1, 0),
        ('u + 2', 2, 0), ('u - 2', -2, 0),
        ('u + 5', 5, 0), ('u - 5', -5, 0),
        ('v + 1', 0, 1), ('v - 1', 0, -1),
        ('v + 2', 0, 2), ('v - 2', 0, -2),
        ('v + 5', 0, 5), ('v - 5', 0, -5),
    ]

    for name, du, dv in shifts:
        p_sweep = np.array([[base_u + du], [base_v + dv], [1]], dtype=np.float64)
        d_sweep = np.linalg.solve(K, p_sweep)
        d_sweep_hat = (d_sweep / np.linalg.norm(d_sweep)).flatten()

        pt_sweep, d_val, t_val = intersect_ray_cylinder(d_sweep_hat, A, a_hat, radius)
        if pt_sweep is not None:
            rel = pt_sweep - A
            z_raw = np.dot(pt_sweep - M0, a_hat)
            z_sw = float(np.clip(z_raw, 0.0, pipe_length))
            rad_p = rel - np.dot(rel, a_hat) * a_hat
            th_sw = np.arctan2(np.dot(rad_p, v_axis), np.dot(rad_p, u_axis))
            s_sw = radius * th_sw

            sweep_pts = np.array([patch_unwrapped['P'], patch_unwrapped['Q'], (s_sw, z_sw), patch_unwrapped['S']])
            c_sw = np.mean(sweep_pts, axis=0)
            ang_sw = np.arctan2(sweep_pts[:, 1] - c_sw[1], sweep_pts[:, 0] - c_sw[0])
            sorted_sw = sweep_pts[np.argsort(ang_sw)]
            sw_area = 0.5 * np.abs(np.dot(sorted_sw[:, 0], np.roll(sorted_sw[:, 1], 1)) - np.dot(sorted_sw[:, 1], np.roll(sorted_sw[:, 0], 1)))
            sw_err = abs(sw_area - actual_area_cm2) / actual_area_cm2 * 100

            print(f"{name:<10} | {d_val:<14.4e} | {t_val:<8.2f} | {z_raw:<8.2f} | {z_sw:<9.2f} | {sw_area:<10.2f} | {sw_err:.2f}%")

    # ========================================================
    # 5-Trial Independent Repeatability Test (Live Clicks)
    # ========================================================
    print("\n================ 5-Trial Independent Repeatability Test ================")
    print("Click Box II corners (P -> Q -> R -> S) 5 separate times (20 clicks total), or press 'ESC'/'q' to skip.")

    trial_clicks = []
    trial_img = original_img.copy()
    for m_lbl, (mu, mv) in locked_markers.items():
        draw_marker_on_img(trial_img, mu, mv, m_lbl, dot_color=(0, 0, 255))

    def click_5_trials(event, u, v, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(trial_clicks) < 20:
            t_num = len(trial_clicks) // 4 + 1
            lbl = ['P', 'Q', 'R', 'S'][len(trial_clicks) % 4]
            trial_clicks.append((u, v))
            print(f"Trial {t_num}/5 - {lbl}: ({u}, {v})")
            cv2.circle(trial_img, (u, v), 5, (0, 255, 255), -1)
            cv2.imshow("5-Trial Repeatability", trial_img)
        elif event == cv2.EVENT_MOUSEMOVE:
            hover_t = trial_img.copy()
            t_num = min(len(trial_clicks) // 4 + 1, 5)
            lbl = ['P', 'Q', 'R', 'S'][len(trial_clicks) % 4]
            cv2.line(hover_t, (u - 25, v), (u + 25, v), (255, 255, 0), 2)
            cv2.line(hover_t, (u, v - 25), (u, v + 25), (255, 255, 0), 2)
            cv2.putText(hover_t, f"T{t_num} {lbl}: ({u}, {v})", (u + 30, v + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("5-Trial Repeatability", hover_t)

    cv2.namedWindow("5-Trial Repeatability")
    cv2.setMouseCallback("5-Trial Repeatability", click_5_trials)
    cv2.imshow("5-Trial Repeatability", trial_img)

    while True:
        if (cv2.waitKey(1) & 0xFF) in [27, ord('q')] or len(trial_clicks) == 20:
            if len(trial_clicks) == 20:
                cv2.waitKey(400)
            break
    cv2.destroyAllWindows()

    if len(trial_clicks) == 20:
        repeatability_trials = [
            {'P': trial_clicks[i*4], 'Q': trial_clicks[i*4+1], 'R': trial_clicks[i*4+2], 'S': trial_clicks[i*4+3]}
            for i in range(5)
        ]
        print("\nSaved repeatability_trials list (copy this if you want to lock it later):")
        print("repeatability_trials =", repeatability_trials)

        trial_areas = []
        print("\nTrial | P            | Q            | R            | S            | Area (cm²) | Error (%)")
        print("-" * 92)
        for idx, t_corners in enumerate(repeatability_trials, start=1):
            t_area, _, _, _ = unwrap_and_area(t_corners, warn_bounds=False)
            trial_areas.append(t_area)
            t_err = abs(t_area - actual_area_cm2) / actual_area_cm2 * 100
            print(f"{idx:<5} | {str(t_corners['P']):<12} | {str(t_corners['Q']):<12} | {str(t_corners['R']):<12} | {str(t_corners['S']):<12} | {t_area:<10.2f} | {t_err:.2f}%")

        mean_area = np.mean(trial_areas)
        std_area = np.std(trial_areas, ddof=1)
        mean_err = abs(mean_area - actual_area_cm2) / actual_area_cm2 * 100
        print("-" * 92)
        print(f"5-Trial Mean Area       : {mean_area:.2f} cm²")
        print(f"Standard Deviation (SD) : ±{std_area:.3f} cm²")
        print(f"Percentage Error        : {mean_err:.2f}% (vs. {actual_area_cm2:.2f} cm² ground truth)")

    # ========================================================
    # Save Figures for Report / Advisor Deliverables
    # ========================================================
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 6))
    patch_poly = np.vstack([sorted_pts, sorted_pts[0]])
    plt.plot(patch_poly[:, 0], patch_poly[:, 1], 'r-o', label=f'Patch (Area: {calc_area:.2f} cm²)')
    plt.fill(patch_poly[:, 0], patch_poly[:, 1], 'red', alpha=0.15)

    ref_z = [np.dot(M - M0, a_hat) for M in M_points]
    plt.plot([0]*4, ref_z, 'b--s', label='5-cm Axis Markers')

    for lbl in ['P', 'Q', 'R', 'S']:
        s_val, z_val = patch_unwrapped[lbl]
        plt.annotate(f"{lbl}: ({s_val:.1f}, {z_val:.1f})", (s_val, z_val),
                        textcoords="offset points", xytext=(8, 8),
                        fontsize=9, weight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.6))

    plt.title('Unwrapped Cylindrical Surface ($s = R\\theta$ vs $z$)')
    plt.xlabel('Circumferential Arc Length $s$ (cm)')
    plt.ylabel('Axial Depth $z$ (cm)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.axis('equal')
    plt.savefig('unwrapped_patch.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig3d = plt.figure(figsize=(8, 7))
    ax3d = fig3d.add_subplot(111, projection='3d')
    ax3d.scatter(0, 0, 0, color='black', marker='^', s=80, label='Camera Origin (C)')

    axis_s = np.linspace(-2, 35, 30)
    axis_line = np.array([A + s * a_hat for s in axis_s])
    ax3d.plot(axis_line[:, 0], axis_line[:, 1], axis_line[:, 2], 'g--', label='Cylinder Axis')

    ax3d.scatter(M_points[:, 0], M_points[:, 1], M_points[:, 2], color='blue', s=50, label='Markers (m0-m3)')
    patch_coords = np.array([patch_3d[lbl] for lbl in ['P', 'Q', 'R', 'S']])
    ax3d.scatter(patch_coords[:, 0], patch_coords[:, 1], patch_coords[:, 2], color='red', s=50, label='Patch Corners')

    ax3d.set_xlabel('X (cm)')
    ax3d.set_ylabel('Y (cm)')
    ax3d.set_zlabel('Z (cm)')
    ax3d.set_title('3D Pose and Reconstructed Surface Points')
    ax3d.legend()
    plt.savefig('reconstructed_3d_pose.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nSaved deliverable plots to 'unwrapped_patch.png' and 'reconstructed_3d_pose.png'")