#!/usr/bin/env bash
set -euo pipefail

REPO="${DINGO_REPO:-kisungyou/Dingo}"
VERSION="${1:-latest}"
ARCH="${2:-}"

if [[ -z "${ARCH}" ]]; then
  case "$(uname -m)" in
    x86_64|amd64) ARCH="x86_64" ;;
    arm64|aarch64) ARCH="arm64" ;;
    *) ARCH="$(uname -m)" ;;
  esac
fi

DEB_ARCH="${ARCH}"
if [[ "${ARCH}" == "x86_64" ]]; then
  DEB_ARCH="amd64"
fi

if [[ "${VERSION}" == "latest" ]]; then
  VERSION="$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" | sed -n 's/.*"tag_name": "v\{0,1\}\([^"]*\)".*/\1/p' | head -n1)"
fi

if [[ -z "${VERSION}" ]]; then
  echo "Unable to resolve release version" >&2
  exit 1
fi

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

if [[ "${OS}" == "darwin" ]]; then
  ASSET="dingo-${VERSION}-macos-${ARCH}.pkg"
  URL="https://github.com/${REPO}/releases/download/v${VERSION}/${ASSET}"
  curl -fL "${URL}" -o "${TMP_DIR}/${ASSET}"
  sudo installer -pkg "${TMP_DIR}/${ASSET}" -target /
  exit 0
fi

if command -v apt >/dev/null 2>&1; then
  ASSET="dingo_${VERSION}_${DEB_ARCH}.deb"
  URL="https://github.com/${REPO}/releases/download/v${VERSION}/${ASSET}"
  curl -fL "${URL}" -o "${TMP_DIR}/${ASSET}"
  sudo apt install -y "${TMP_DIR}/${ASSET}"
  exit 0
fi

if command -v dnf >/dev/null 2>&1 || command -v yum >/dev/null 2>&1 || command -v zypper >/dev/null 2>&1; then
  ASSET="dingo-${VERSION}-1.${ARCH}.rpm"
  URL="https://github.com/${REPO}/releases/download/v${VERSION}/${ASSET}"
  curl -fL "${URL}" -o "${TMP_DIR}/${ASSET}"
  if command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y "${TMP_DIR}/${ASSET}"
  elif command -v yum >/dev/null 2>&1; then
    sudo yum install -y "${TMP_DIR}/${ASSET}"
  else
    sudo zypper --non-interactive install "${TMP_DIR}/${ASSET}"
  fi
  exit 0
fi

ASSET="dingo-${VERSION}-linux-${ARCH}.tar.gz"
URL="https://github.com/${REPO}/releases/download/v${VERSION}/${ASSET}"
curl -fL "${URL}" -o "${TMP_DIR}/${ASSET}"
sudo tar -xzf "${TMP_DIR}/${ASSET}" -C /
