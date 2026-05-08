"""Hermes Memory UI plugin package.

This repository is primarily a dashboard plugin. Hermes' generic plugin
loader also imports enabled plugins from their repository root, so expose a
minimal no-op register() hook to keep that loader quiet while the dashboard
runtime mounts dashboard/plugin_api.py separately.
"""


def register(ctx):
    """Register root-level Hermes extensions.

    The memory UI currently provides only dashboard assets and API routes, so
    there are no root-level tools, commands, or hooks to register.
    """
    return None
