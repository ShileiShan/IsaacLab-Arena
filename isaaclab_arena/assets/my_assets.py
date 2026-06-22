import os

from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.register import register_asset
import isaaclab.sim as sim_utils

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@register_asset
class MaterialBox003Kinematic(LibraryObject):
    name = "material_box_003_kinematic"
    tags = ["object", "fixture", "destination"]
    usd_path = os.path.join(_REPO_ROOT, "assets", "MaterialBox003", "MaterialBox003.usd")
    object_type = ObjectType.RIGID
    spawn_cfg_addon = {
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        "collision_props": sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    }
