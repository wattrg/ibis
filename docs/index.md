---
title: Ibis
layout: splash
subtitle: Performance portable computational fluid dynamics
excerpt: "Performance protable computational fluid dynamics solver"
header:
  overlay_color: "#000"
  overlay_filter: "0.5"
  overlay_image: "/assets/images/supersonic_jet_test.png"
  actions:
    - label: "Get Started"
      url: "/docs/getting_started/install"
    - label: "Github"
      url: "https://github.com/wattrg/ibis"

feature_row:
  - image_path: /assets/images/open-lock.png
    alt: "open source"
    title: "Open Source"
    excerpt: "Ibis is completely open source and free to use"
  - image_path: /assets/images/speedo-icon.png
    alt: "Performance portable"
    title: "Performance Portable"
    excerpt: "Ibis leverages Kokkos to run effeciently on modern HPC"
    url: "https://kokkos.org"
    btn_label: "Learn More"
    btn_class: "btn--primary"
  - image_path: /assets/images/flexibility.png
    alt: "Flexibility"
    title: "Flexibility"
    excerpt: "Unstructured grids allow for complex flows to be simulated easily"

intro:
  - excerpt: "Compressible computational fluid dynamics for modern HPC"
---

{% include feature_row id="intro" type="center" %}

{% include feature_row %}
