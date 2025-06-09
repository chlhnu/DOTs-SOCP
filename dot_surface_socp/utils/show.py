import numpy as np
import pyvista as pv
import imageio.v2 as imageio
import logging
from dot_surface_socp.config import LOG_LEVELS
from pathlib import Path
from typing import Union
from dot_surface_socp.utils.type import CameraConfig

# =======================================
# Available colormaps for PyVista
# =======================================
# 	Note: PyVista uses Matplotlib colormaps, so you can use any colormap available in Matplotlib.
# 		  The following are some common colormaps:
# 	Note: "_r" is used to reverse the colormap
AVAILABLE_COLORMAPS = [
	"bone_r", "gist_heat_r", "GnBu", "YlGnBu", "Blues"
]
DEFAULT_CMAP_SAVE = AVAILABLE_COLORMAPS[0]
DEFAULT_CMAP_SHOW = AVAILABLE_COLORMAPS[2]

CLIM_MAX = 1.0 # Maximum value for the densities to plot

# =======================================
# Visualization tools
# =======================================

def create_pv_mesh(vertices, triangles):
	"""Create a PyVista mesh from vertices and triangles.
	"""
	faces = np.column_stack((np.ones(triangles.shape[0], dtype=np.int64) * 3, triangles))
	return pv.PolyData(vertices, faces.flatten())

def set_camera_with_config(plotter: pv.plotter, camera_config: Union[None, CameraConfig, list[CameraConfig]]):
	"""Set camera position using a configuration dictionary.
	"""

	if isinstance(camera_config, dict):
		_camera_config = camera_config
	elif isinstance(camera_config, list):
		_camera_config = camera_config[0]
	elif camera_config is None:
		plotter.view_isometric()
		return
	else:
		raise ValueError("camera_config should be either None, a dictionary, or a list of dictionaries")

	assert isinstance(_camera_config, dict), \
		"(every) camera_config should be a dictionary containing camera settings"
	assert all(key in _camera_config for key in ["position", "focal_point", "up"]), \
		"(every) camera_config should contain 'position', 'focal_point', and 'up' keys"
	
	_camera = pv.Camera()
	_camera.position = _camera_config["position"]
	_camera.focal_point = _camera_config["focal_point"]
	_camera.up = _camera_config["up"]
	plotter.camera = _camera


def __render_mesh(mesh, scalar_data=None, title=None, 
				show=False, cmap=DEFAULT_CMAP_SAVE, 
				background_color='white', window_size=(1200, 1200)):
	"""The rendering function to make frames (render a single frame with only one camera configuration).
	"""
	options_dict = {
		"smooth_shading": True, # Smooth shading
		"specular": 0.5,		# Specular reflection intensity  
		"specular_power": 15,	# Specular reflection focus
		"show_scalar_bar": False,
	}

	plotter = pv.Plotter(window_size=window_size, off_screen=not show)
	plotter.background_color = background_color
	
	if scalar_data is not None:
		mesh["values"] = scalar_data
		plotter.add_mesh(
			mesh, 
			scalars="values",
			show_edges=False,
			cmap=cmap,
			clim=[0.0, CLIM_MAX],
			**options_dict
		)
	else:
		plotter.add_mesh(
			mesh,
			color='white',
			show_edges=True,
			line_width=0.5,
			**options_dict
		)
	
	if title:
		plotter.add_text(title, font_size=14, position="upper_edge")
	
	return plotter

def render_mesh(mesh, scalar_data=None, title=None, 
				show=False, save_path=None, cmap=None, 
				camera_config=None, background_color=None, window_size=None):
	"""The rendering function to make frames.

	Render a mesh using pyvista, either with scalar data visualization or mesh structure.
	This function provides two main rendering modes:
		1. When scalar_data is provided: Renders the mesh with scalar values mapped to colors
		2. When scalar_data is None: Renders the mesh structure with white faces and edges
	
	Parameters:
	----------
		mesh : pyvista.PolyData or pyvista.UnstructuredGrid
			The mesh to be rendered
		scalar_data : array-like, optional
			Scalar values to be mapped onto the mesh surface. If None, renders mesh structure (default: None)
		title : str, optional
			Title to display on the plot (default: None)
		show : bool, optional
			Whether to display the plot interactively (default: False)
		save_path : str, optional
			Path to save the rendered image. If None, image won't be saved (default: None)
		cmap : str, optional
			Colormap for scalar data visualization
		camera_config : dict, optional
			configurations for camera position (default: None)
		background_color : str, optional
			Color of the plot background (default: 'white')
		window_size : tuple, optional
			Size of the rendering window in pixels (default: (1200, 1200))

	Examples:
	--------
	# Render mesh structure
	>>> render_mesh(mesh)
	# Render mesh with scalar data
	>>> render_mesh(mesh, scalar_data=density_values)
	"""

	assert not (show and save_path), "Cannot specify both 'show' and 'save_path'. Choose one."
	assert show or save_path, "Must specify either 'show' or 'save_path'."

	kwargs = {
		"mesh": mesh,
		"scalar_data": scalar_data,
		"title": title,
		"show": show,
		"cmap": cmap,
		"background_color": background_color,
		"window_size": window_size
	}
	_save_path = None

	if save_path:
		_save_path = save_path
		if isinstance(camera_config, list):
			save_path = Path(save_path)
			path, name, ext = save_path.parent, save_path.stem, save_path.suffix

			_save_path = []
			for num_config, config in enumerate(camera_config):
				plotter = __render_mesh(**kwargs)
				camera_id = config["name"] if "name" in config else f"camera_{num_config:02d}"
				filename_with_camera_id = path / f"{name}_{camera_id}{ext}"
				set_camera_with_config(plotter, config)
				plotter.screenshot(filename_with_camera_id, transparent_background=False, return_img=False)
				_save_path.append(filename_with_camera_id)
				plotter.close()
		else:
			plotter = __render_mesh(**kwargs)
			set_camera_with_config(plotter, camera_config)
			plotter.screenshot(save_path, transparent_background=False, return_img=False)
			plotter.close()
	elif show:
		plotter = __render_mesh(**kwargs)
		set_camera_with_config(plotter, camera_config)
		plotter.show()
		plotter.close()
	else:
		raise RuntimeError("Unexpected state: neither 'show' nor 'save_path' is valid despite initial assertions")
	
	return _save_path

def save_description_of_dot(mesh, mu0, mu1, filename_format_example, camera_config = None, cmap = None):
	"""Save the pictures of the mesh structure and the initial and final densities.
	"""
	if cmap is None:
		cmap = DEFAULT_CMAP_SAVE
	
	# The mesh structure
	render_mesh(
		mesh, None,
		# title=f"Mesh Structure - {example_name}",
		save_path=filename_format_example.format(description="mesh"),
		camera_config=camera_config
	)
	
	# The initial densities
	render_mesh(
		mesh, mu0,
		# title=f"Initial Density - {example_name}",
		save_path=filename_format_example.format(description="mu0"),
		camera_config=camera_config,
		cmap=cmap
	)
	
	# The final densities
	render_mesh(
		mesh, mu1,
		# title=f"Final Density - {example_name}",
		save_path=filename_format_example.format(description="mu1"),
		camera_config=camera_config,
		cmap=cmap
	)

def save_results_of_dot(mesh, mu, filename_format_animation, num_frames = 5, camera_config = None, cmap = None):
	if cmap is None:
		cmap = DEFAULT_CMAP_SAVE
	
	# Show the evolution of the densities
	n_time = mu.shape[0]
	selected_frames = [int(i * (n_time - 1) / (num_frames - 1)) for i in range(num_frames)]
	for i in selected_frames:
		render_mesh(
			mesh, mu[i,:],
			# title=f"Time Step {i+1}/{n_time} - {example_name}",
			save_path=filename_format_animation.format(time_frame_number=i+1),
			camera_config=camera_config,
			cmap=cmap
		)

def save_animation(mesh, to_plot, animation_filename: str, window_size = (1200, 1200), camera_config = None, cmap = None):
	"""Saves an animation of the mesh with density evolution.
	
	Args:
		mesh: PyVista mesh object
		to_plot: Array of density values over time (n_time x n_vertices) 
		animation_filename: Path to save the animation MP4 file
		n_time: Number of time steps
		camera_config: Optional camera configuration
		example_name: Optional name of the example
	"""
	if cmap is None:
		cmap = DEFAULT_CMAP_SAVE
	
	# Make a temporary directory for the frames
	temp_dir = Path(animation_filename).parent / "temp_frames"
	temp_dir.mkdir(exist_ok=True)
	n_time = to_plot.shape[0]
	n_camera = len(camera_config) if isinstance(camera_config, list) else 1
	
	logging.log(LOG_LEVELS['info'], "Generating animation frames...")
	frame_files = []
	for idx in range(n_time):
		frame_path = temp_dir / f"frame_{idx:04d}.png"
		saved_frame_path = render_mesh(
			mesh, to_plot[idx,:],
			# title=f"Time Step {i+1}/{n_time} - {example_name}",
			show=False,
			save_path=frame_path,
			camera_config=camera_config,
			window_size=window_size,
			cmap=cmap
		)

		# if n_camera > 1:
		if saved_frame_path and isinstance(saved_frame_path, list):
			saved_frame_path = [path for path in saved_frame_path]
		frame_files.append(saved_frame_path)

		# Log progress
		if (idx+1) % 5 == 0 or idx == n_time-1:
			logging.log(LOG_LEVELS['info'], f"Generated {idx+1}/{n_time} frames")

	# Transpose nested list
	# if n_camera > 1:
	if frame_files \
			and isinstance(frame_files[0], list) \
			and len(set([len(list(sublist)) for sublist in frame_files])) == 1:
		frame_files = list(map(list, zip(*frame_files)))
	
	# Create the animation
	def create_animation(_filename, _frame_files):
		logging.log(LOG_LEVELS['info'], f"Creating animation at {_filename}...")
		writer = imageio.get_writer(_filename, fps=10)
		for file in _frame_files:
			writer.append_data(imageio.imread(file))
		writer.close()
	
	if isinstance(frame_files[0], list):
		_animation_filename = Path(animation_filename)
		path, name, ext = _animation_filename.parent, _animation_filename.stem, _animation_filename.suffix

		for idx, camera_frames in enumerate(frame_files):
			try:
				camera_id = camera_config[idx]["name"]
			except:
				camera_id = f"camera_{idx:02d}"
			
			filename_with_camera_id = path / f"{name}_{camera_id}{ext}"
			create_animation(filename_with_camera_id, camera_frames)
	else:
		create_animation(animation_filename, frame_files)
	
	# Remove temporary files
	flat_frame_files = [item for sublist in frame_files for item in sublist] if isinstance(frame_files[0], list) else frame_files
	for file in flat_frame_files:
		try:
			file.unlink()
		except Exception as e:
			logging.warning(f"Could not remove temp file {file}: {e}")
	try:
		temp_dir.rmdir()
	except Exception as e:
		logging.warning(f"Could not remove temp directory {temp_dir}: {e}")
	
	logging.log(LOG_LEVELS['info'], "Animation created successfully")
	
def show_animation(mesh, to_plot, example_name: str = None, window_size = (1200, 1200), camera_config = None, cmap = None):
	"""Shows an interactive animation of the mesh with density evolution.
	
	Args:
		mesh: PyVista mesh object
		to_plot: Array of density values over time (n_time x n_vertices)
		example_name: Name of the example being shown
		window_size: Tuple of (width, height) for the window
		camera_config: Optional camera configuration
		cmap: Colormap for the density values
	"""
	if cmap is None:
		cmap = DEFAULT_CMAP_SHOW
	
	logging.log(LOG_LEVELS['info'], "Showing animation. Use LEFT/RIGHT arrow keys to navigate frames, Q to exit...")
	plotter = pv.Plotter(window_size=window_size)
	plotter.background_color = 'white'
	
	frame = [0]
	n_time = to_plot.shape[0]
	
	# Initial frame
	mesh["values"] = to_plot[0,:]
	plotter.add_mesh(
		mesh, 
		scalars="values",
		cmap=cmap,
		clim=[0.0, CLIM_MAX],
		smooth_shading=True,
		specular=0.5,
		specular_power=15,
		show_edges=True,
		line_width=0.1,
		show_scalar_bar=False,
	)
	
	title_text = f"Time Step 1/{n_time} - {example_name}" if example_name else f"Time Step 1/{n_time}"
	title = plotter.add_text(title_text, font_size=14, position="upper_edge")
	
	# Set camera
	set_camera_with_config(plotter, camera_config)
	
	slider_widget = [None]
	
	def next_frame():
		frame[0] = (frame[0] + 1) % n_time
		mesh["values"] = to_plot[frame[0],:]
		plotter.remove_actor(title)
		new_text = f"Time Step {frame[0]+1}/{n_time} - {example_name}" if example_name else f"Time Step {frame[0]+1}/{n_time}"
		plotter.add_text(new_text, font_size=14, position="upper_edge", name="title")
		
		if slider_widget[0] is not None:
			slider_widget[0].GetRepresentation().SetValue(100 * frame[0] / (n_time - 1))
			
		plotter.render()
	
	def prev_frame():
		frame[0] = (frame[0] - 1) % n_time
		mesh["values"] = to_plot[frame[0],:]
		plotter.remove_actor(title)
		new_text = f"Time Step {frame[0]+1}/{n_time} - {example_name}" if example_name else f"Time Step {frame[0]+1}/{n_time}"
		plotter.add_text(new_text, font_size=14, position="upper_edge", name="title")
		
		if slider_widget[0] is not None:
			slider_widget[0].GetRepresentation().SetValue(100 * frame[0] / (n_time - 1))
			
		plotter.render()
	
	# The control keys
	plotter.add_key_event('Right', next_frame)
	plotter.add_key_event('Left', prev_frame)
	plotter.add_key_event('d', next_frame)
	plotter.add_key_event('a', prev_frame)
	
	def update_slider(value):
		frame_idx = int((value/100) * (n_time - 1))
		frame[0] = frame_idx
		mesh["values"] = to_plot[frame_idx,:]
		plotter.remove_actor(title)
		new_text = f"Time Step {frame_idx+1}/{n_time} - {example_name}" if example_name else f"Time Step {frame_idx+1}/{n_time}"
		plotter.add_text(new_text, font_size=14, position="upper_edge", name="title")
		return
	
	slider_widget[0] = plotter.add_slider_widget(
		update_slider,
		[0, 100],
		title="Frame",
		title_height=0.01,
		title_opacity=0.75,
		title_color="black",
		value=0,
		pointa=(0.3, 0.02),
		pointb=(1.0, 0.02),
		style="modern",
		color="lightblue",
		fmt="%.0f%%",
		interaction_event="always"
	)
	
	# Show
	plotter.add_text("Use LEFT/RIGHT or A/D keys to navigate frames\n"
			"Or use the slider right\n"
			"Press Q to exit", 
			position="lower_left", font_size=12, color="black")
	try:
		plotter.show()
	except Exception as e:
		logging.error(f"Error during plotter.show(): {e}")

# -----------------------------------------------------------------------------------------------
# Normalization of density to plot
# -----------------------------------------------------------------------------------------------
from dot_surface_socp.utils.type import GeometryData
from dot_surface_socp.utils.util import translate_density

def decorator_factory_power_perceptual(power: float = None):
	"""Decorator factory to apply a power transformation to density values for better visualization.

	Args:
        power (float): The exponent for the power-law transformation.
		If None, no transformation is applied, i.e. power == 1.0.

    Returns:
        function: A decorator.
	"""
	if power is None:
		power = 1.0

	def apply_power(data: np.ndarray) -> np.ndarray:
		return CLIM_MAX * np.power(np.maximum(data, 0.0), power) / np.power(CLIM_MAX, power)

	def decorator(func):
		def wrapper(*args, **kwargs):
			result = func(*args, **kwargs)
			return tuple(apply_power(r) for r in result)
		return wrapper
	
	return decorator

def normalize_density_to_plot(mu: np.ndarray, geometry: GeometryData):
	"""Normalize the densities of transpotation to plot.
	"""
	mu_normalized = translate_density(mu, geometry)
	mu0_normalized = translate_density(geometry["mu0"], geometry)
	mu1_normalized = translate_density(geometry["mu1"], geometry)

	max_val = max([np.max(mu0_normalized), np.max(mu1_normalized)])
	# max_val = max(max_val, np.max(mu_normalized))
	to_plot = np.clip(CLIM_MAX / max_val * mu_normalized, - np.inf, CLIM_MAX)
	to_plot_mu0 = np.clip(CLIM_MAX / max_val * mu0_normalized, - np.inf, CLIM_MAX)
	to_plot_mu1 = np.clip(CLIM_MAX / max_val * mu1_normalized, - np.inf, CLIM_MAX)

	return to_plot, to_plot_mu0, to_plot_mu1

def normalize_density_to_plot2(mu: np.ndarray, geometry: GeometryData):
	"""Normalize the densities of transpotation to plot.
	"""
	mu_normalized = translate_density(mu, geometry)
	mu0_normalized = translate_density(geometry["mu0"], geometry)
	mu1_normalized = translate_density(geometry["mu1"], geometry)

	def __normalize_data(data: np.ndarray) -> np.ndarray:
		if len(data.shape) == 1:
			return data / np.max(data)
		elif len(data.shape) == 2:
			size_time = data.shape[0]
			normalized_data = np.zeros_like(data)
			for t in range(size_time):
				normalized_data[t, :] = CLIM_MAX / np.max(data[t, :]) * data[t, :]
			return normalized_data
		else:
			raise ValueError("The input data should be 1D or 2D array.")

	to_plot = __normalize_data(mu_normalized)
	to_plot_mu0 = __normalize_data(mu0_normalized)
	to_plot_mu1 = __normalize_data(mu1_normalized)

	return to_plot, to_plot_mu0, to_plot_mu1
