"""Keep repository-specific VCS rules out of source distributions."""

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        # Hatchling force-includes VCS ignore files even with an explicit file allowlist.
        for source in list(build_data["force_include"]):
            if Path(source).name in (".gitignore", ".hgignore"):
                del build_data["force_include"][source]
