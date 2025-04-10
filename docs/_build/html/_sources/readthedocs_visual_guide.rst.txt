ReadTheDocs Visual Guide
=====================

This visual guide provides screenshots and step-by-step instructions for using the ReadTheDocs platform to deploy your JAX-bandflux documentation.

ReadTheDocs Dashboard
-------------------

When you log in to ReadTheDocs, you'll see a dashboard similar to this:

.. code-block:: text

    +-----------------------------------------------------------------------+
    |                                                                       |
    |  ReadTheDocs Dashboard                                                |
    |                                                                       |
    |  +-------------------+  +----------------------------------------+    |
    |  |                   |  |                                        |    |
    |  | My Projects       |  | Recent Projects                        |    |
    |  |                   |  |                                        |    |
    |  | + Import Project  |  | JAX-bandflux                           |    |
    |  |                   |  | Last built: 2 hours ago                |    |
    |  | JAX-bandflux      |  | Status: Passing                        |    |
    |  | Other Project     |  |                                        |    |
    |  |                   |  | View Docs | Build | Admin              |    |
    |  +-------------------+  +----------------------------------------+    |
    |                                                                       |
    +-----------------------------------------------------------------------+

Importing a Project
-----------------

To import a new project, click on "Import Project" and you'll see a screen like this:

.. code-block:: text

    +-----------------------------------------------------------------------+
    |                                                                       |
    |  Import a Project                                                     |
    |                                                                       |
    |  Connect to GitHub                 [Connected]                        |
    |                                                                       |
    |  Filter repositories:  [Search box                       ]            |
    |                                                                       |
    |  +-----------------------------------------------------------+        |
    |  |                                                           |        |
    |  | samleeney/JAX-bandflux                                    |        |
    |  | A JAX-based package for calculating supernovae Bandfluxes |        |
    |  |                                           [Import] button |        |
    |  |                                                           |        |
    |  +-----------------------------------------------------------+        |
    |                                                                       |
    +-----------------------------------------------------------------------+

Project Configuration
------------------

After importing, you'll need to configure your project:

.. code-block:: text

    +-----------------------------------------------------------------------+
    |                                                                       |
    |  Project Details                                                      |
    |                                                                       |
    |  Name: JAX-bandflux                                                   |
    |  Repository URL: https://github.com/samleeney/JAX-bandflux           |
    |  Repository type: Git                                                 |
    |  Default branch: main                                                 |
    |                                                                       |
    |  Advanced Settings:                                                   |
    |                                                                       |
    |  Documentation type: [Sphinx] ▼                                       |
    |  Programming language: [Python] ▼                                     |
    |  Python configuration file: [docs/conf.py]                            |
    |  Python interpreter: [CPython 3.x] ▼                                  |
    |                                                                       |
    |  [Save] button                                                        |
    |                                                                       |
    +-----------------------------------------------------------------------+

Project Dashboard
--------------

Once your project is set up, you'll see the project dashboard:

.. code-block:: text

    +-----------------------------------------------------------------------+
    |                                                                       |
    |  JAX-bandflux                                                         |
    |                                                                       |
    |  Overview | Downloads | Versions | Builds | Admin                     |
    |                                                                       |
    |  Project Description:                                                 |
    |  A JAX-based package for calculating supernovae Bandfluxes            |
    |                                                                       |
    |  Latest Documentation: https://jax-bandflux.readthedocs.io/          |
    |                                                                       |
    |  Last Built: 10 minutes ago                                           |
    |  Build Status: Passing                                                |
    |                                                                       |
    |  [View Docs] [Build Version: latest] buttons                          |
    |                                                                       |
    +-----------------------------------------------------------------------+

Build Page
--------

The build page shows the status of your documentation builds:

.. code-block:: text

    +-----------------------------------------------------------------------+
    |                                                                       |
    |  Builds for JAX-bandflux                                              |
    |                                                                       |
    |  +-----------------------------------------------------------+        |
    |  | Version: latest                                           |        |
    |  | Date: 2025-10-04 14:30:45                                 |        |
    |  | Success                                                   |        |
    |  | Triggered by: GitHub webhook                              |        |
    |  |                                                           |        |
    |  | [View Docs] [View Build]                                  |        |
    |  +-----------------------------------------------------------+        |
    |                                                                       |
    |  +-----------------------------------------------------------+        |
    |  | Version: latest                                           |        |
    |  | Date: 2025-10-04 13:15:22                                 |        |
    |  | Failed                                                    |        |
    |  | Triggered by: GitHub webhook                              |        |
    |  |                                                           |        |
    |  | [View Build]                                              |        |
    |  +-----------------------------------------------------------+        |
    |                                                                       |
    +-----------------------------------------------------------------------+

Build Log
-------

When you click "View Build", you'll see the build log:

.. code-block:: text

    +-----------------------------------------------------------------------+
    |                                                                       |
    |  Build Log for JAX-bandflux (latest)                                  |
    |                                                                       |
    |  Command: git clone --depth 1 https://github.com/samleeney/JAX-bandflux |
    |  Cloning into 'JAX-bandflux'...                                       |
    |  ...                                                                  |
    |                                                                       |
    |  Command: python -m pip install --upgrade --no-cache-dir pip setuptools |
    |  ...                                                                  |
    |                                                                       |
    |  Command: python -m pip install --upgrade --no-cache-dir -e .[dev]    |
    |  ...                                                                  |
    |                                                                       |
    |  Command: python -m sphinx -T -E -b html -d _build/doctrees -D language=en docs _build/html |
    |  ...                                                                  |
    |                                                                       |
    |  Build successful!                                                    |
    |                                                                       |
    +-----------------------------------------------------------------------+

Versions Page
----------

The versions page allows you to manage which versions of your documentation are built:

.. code-block:: text

    +-----------------------------------------------------------------------+
    |                                                                       |
    |  Versions for JAX-bandflux                                            |
    |                                                                       |
    |  Active Versions:                                                     |
    |                                                                       |
    |  [x] latest (main) - Default                                          |
    |  [x] v0.1.91 (tag)                                                    |
    |  [ ] stable (tag)                                                     |
    |                                                                       |
    |  Inactive Versions:                                                   |
    |                                                                       |
    |  [ ] develop (branch)                                                 |
    |  [ ] v0.1.90 (tag)                                                    |
    |                                                                       |
    |  [Activate] button                                                    |
    |                                                                       |
    +-----------------------------------------------------------------------+

Admin Page - Integrations
-----------------------

The integrations tab shows your connected services:

.. code-block:: text

    +-----------------------------------------------------------------------+
    |                                                                       |
    |  Integrations for JAX-bandflux                                        |
    |                                                                       |
    |  GitHub Integration                                                   |
    |  Status: Active                                                       |
    |  Connected to: samleeney/JAX-bandflux                                 |
    |  Webhook URL: https://readthedocs.org/api/v2/webhook/jax-bandflux/... |
    |                                                                       |
    |  [Sync GitHub Webhook] [Disconnect] buttons                           |
    |                                                                       |
    +-----------------------------------------------------------------------+

Admin Page - Advanced Settings
---------------------------

The advanced settings page allows you to configure build behaviors:

.. code-block:: text

    +-----------------------------------------------------------------------+
    |                                                                       |
    |  Advanced Settings for JAX-bandflux                                   |
    |                                                                       |
    |  Default version: [latest] ▼                                          |
    |                                                                       |
    |  [ ] Show version warning                                             |
    |  [ ] Make project public                                              |
    |  [x] Build pull requests for this project                             |
    |  [x] Show build on pull request                                       |
    |                                                                       |
    |  Privacy Level: [Public] ▼                                            |
    |                                                                       |
    |  [Save] button                                                        |
    |                                                                       |
    +-----------------------------------------------------------------------+

Published Documentation
--------------------

Once your documentation is built successfully, it will be available at a URL like:
https://jax-bandflux.readthedocs.io/

The documentation will include a version selector dropdown in the bottom-left corner of the page, allowing users to switch between different versions of your documentation.

GitHub Webhook Settings
--------------------

In your GitHub repository settings, you'll see the ReadTheDocs webhook:

.. code-block:: text

    +-----------------------------------------------------------------------+
    |                                                                       |
    |  GitHub Repository Webhooks                                           |
    |                                                                       |
    |  +-----------------------------------------------------------+        |
    |  | Webhook: https://readthedocs.org/api/v2/webhook/jax-bandflux/... |  |
    |  | Events: Push, Pull request                                 |        |
    |  | Status: Active                                             |        |
    |  | Last delivery: 10 minutes ago - Successful                 |        |
    |  |                                                           |        |
    |  | [Edit] [Delete]                                           |        |
    |  +-----------------------------------------------------------+        |
    |                                                                       |
    +-----------------------------------------------------------------------+

This visual guide should help you navigate the ReadTheDocs platform and successfully deploy your JAX-bandflux documentation.