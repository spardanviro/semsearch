#!/bin/sh
# Build the installers attached to a GitHub release: the same scripts, with the version they
# install pinned to the release tag (the copies in the repository install `main`).
#
#   sh scripts/release-assets.sh v0.1.0 dist/release
#   gh release create v0.1.0 dist/release/install.ps1 dist/release/install.sh ...
set -eu

tag=${1:?usage: release-assets.sh <tag like v1.2.3> <output folder>}
out=${2:?usage: release-assets.sh <tag like v1.2.3> <output folder>}
case $tag in
    v[0-9]*.[0-9]*.[0-9]*) ;;
    *) echo "tag must look like v1.2.3, got: $tag" >&2; exit 1 ;;
esac
case $tag in
    *[!0-9A-Za-z.-]*) echo "tag has characters a script cannot hold safely: $tag" >&2; exit 1 ;;
esac

here=$(dirname -- "$0")
mkdir -p "$out"
sed "s/^    \$defaultRef = 'main'/    \$defaultRef = '$tag'/" "$here/install.ps1" >"$out/install.ps1"
# LF only: a Windows checkout may hold CRLF, and `sh` on Linux fails on the `\r`
tr -d '\r' <"$here/install.sh" | sed "s/^DEFAULT_REF=main/DEFAULT_REF=$tag/" >"$out/install.sh"

grep -q "^    \$defaultRef = '$tag'" "$out/install.ps1" \
    && grep -q "^DEFAULT_REF=$tag " "$out/install.sh" \
    || { echo "could not pin the version in the installers" >&2; exit 1; }
echo "wrote $out/install.ps1 and $out/install.sh (installing $tag)"
