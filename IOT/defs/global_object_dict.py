from typing import Optional, TypeVar, Dict, Type, cast

T = TypeVar("T")

# Global instance used to hold the objects
global_dict: Dict[str, object] = {}


class GlobalObjectDict:
	"""
	Defines a global dict that holds object instances.
	Each instance is identified by a name, and has an associated type.

	This class is useful to create objects that can be passed to subprocesses right as they are spawned. This is
	necessary on systems that create new processes using the "spawn" method.
	At runtime, each subprocess will have its own global dict, but the objects contained within will all point to the
	objects in the dict of the parent process, provided that the parent passed GlobalObjectDict.set as the method to
	run to init the subprocess (either in the Process constructor or through an intermediate class, such as the
	initializer argument in the ProcessPoolExecutor or Pool constructor).
	"""

	@staticmethod
	def get(name: str, obj_type: Type[T]) -> Optional[T]:
		"""
		Returns the object identified by the given name from the global dict, or none if the global dict does not
		contain an object identified by that name.
		"""
		global global_dict
		try:
			return cast(obj_type, global_dict[name])
		except KeyError:
			return None

	@staticmethod
	def set(name: str, obj_type: Type[T], value: T):
		"""
		Adds an object to the global dict identified by the given name. If the object already exits, it will be
		overwritten.
		"""
		global global_dict
		assert type(value) == obj_type, "Incorrect parameter type"
		global_dict[name] = value
