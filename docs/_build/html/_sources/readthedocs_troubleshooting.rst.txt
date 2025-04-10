ReadTheDocs Troubleshooting Guide
============================

This guide addresses common issues you might encounter when deploying JAX-bandflux documentation to ReadTheDocs and provides solutions to resolve them.

Table of Contents
----------------

1. `Build Failures`_
2. `Documentation Rendering Issues`_
3. `Webhook and Integration Problems`_
4. `Version Control Issues`_
5. `Custom Domain Problems`_
6. `PDF/ePub Generation Issues`_

Build Failures
------------

Missing Dependencies
~~~~~~~~~~~~~~~~~~

**Problem**: Build fails with errors about missing Python packages.

**Solution**:

1. Check your ``.readthedocs.yml`` file to ensure it correctly specifies how to install dependencies:

   .. code-block:: yaml

      python:
        install:
          - method: pip
            path: .
            extra_requirements:
              - dev

2. Verify that all documentation dependencies are included in your ``setup.py`` file under the 'dev' extras:

   .. code-block:: python

      extras_require={
          'dev': [
              'sphinx>=4.0.0',
              'sphinx-rtd-theme>=1.0.0',
              'sphinxcontrib-mermaid>=0.7.0',
              # other dependencies
          ],
      }

3. If using a requirements file, ensure it includes all necessary packages.

Import Errors
~~~~~~~~~~~~

**Problem**: Build fails with errors when importing your package modules.

**Solution**:

1. Make sure your package is installed during the documentation build:

   .. code-block:: yaml

      python:
        install:
          - method: pip
            path: .

2. Check if your package has C extensions or compiled components that might not build on ReadTheDocs:
   
   - You may need to mock these modules in your ``conf.py``:
   
     .. code-block:: python
     
         autodoc_mock_imports = ['module_with_c_extensions']

3. Verify that your package works with the Python version specified in your ``.readthedocs.yml``.

Sphinx Extension Issues
~~~~~~~~~~~~~~~~~~~~~

**Problem**: Build fails with errors related to Sphinx extensions.

**Solution**:

1. Ensure all required Sphinx extensions are installed:

   .. code-block:: python

      # In setup.py
      extras_require={
          'dev': [
              'sphinx>=4.0.0',
              'sphinx-rtd-theme>=1.0.0',
              'sphinxcontrib-mermaid>=0.7.0',
              # other extensions
          ],
      }

2. Check for compatibility issues between extension versions:
   
   - Some extensions may not work with the latest Sphinx version
   - You might need to pin specific versions

3. Verify that extensions are correctly configured in ``conf.py``.

Memory Errors
~~~~~~~~~~~

**Problem**: Build fails with memory errors for large documentation projects.

**Solution**:

1. Optimize your documentation build:
   
   - Reduce the number of files processed at once
   - Use selective imports in autodoc

2. Contact ReadTheDocs support for projects with genuinely high memory requirements.

Documentation Rendering Issues
----------------------------

Broken Links
~~~~~~~~~~

**Problem**: Documentation contains broken internal or external links.

**Solution**:

1. Use the linkcheck builder to find broken links:

   .. code-block:: bash

      cd docs
      make linkcheck

2. Fix any broken internal links:
   
   - Check for typos in references
   - Ensure referenced files exist

3. For external links:
   
   - Update or remove outdated links
   - Consider using link checkers in your CI pipeline

Missing or Incorrect Content
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Some documentation content is missing or incorrectly rendered.

**Solution**:

1. Build documentation locally to identify issues:

   .. code-block:: bash

      cd docs
      make html

2. Check for RST syntax errors:
   
   - Incorrect indentation
   - Missing blank lines between sections
   - Malformed directives

3. Verify that all source files are included in your table of contents.

Theme and Styling Issues
~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Documentation doesn't look as expected or has styling problems.

**Solution**:

1. Check your theme configuration in ``conf.py``:

   .. code-block:: python

      html_theme = 'sphinx_rtd_theme'
      html_theme_options = {
          # theme options
      }

2. Verify that custom CSS is properly included:

   .. code-block:: python

      html_static_path = ['_static']
      html_css_files = ['custom.css']

3. Test with different browsers to identify browser-specific issues.

Mermaid Diagram Issues
~~~~~~~~~~~~~~~~~~~~

**Problem**: Mermaid diagrams don't render correctly.

**Solution**:

1. Ensure the ``sphinxcontrib.mermaid`` extension is installed and configured:

   .. code-block:: python

      extensions = [
          # other extensions
          'sphinxcontrib.mermaid',
      ]

2. Check your Mermaid syntax for errors

3. Verify that the JavaScript is loading correctly in the built documentation

Webhook and Integration Problems
------------------------------

Webhook Not Triggering Builds
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Pushing to GitHub doesn't trigger a new documentation build.

**Solution**:

1. Check webhook status in GitHub:
   
   - Go to your repository settings > Webhooks
   - Look for the ReadTheDocs webhook
   - Check recent deliveries for errors

2. Verify webhook configuration in ReadTheDocs:
   
   - Go to Admin > Integrations
   - Check that the GitHub integration is active

3. Manually sync the webhook:
   
   - In ReadTheDocs, go to Admin > Integrations
   - Click "Sync GitHub Webhook"

Authentication Issues
~~~~~~~~~~~~~~~~~~

**Problem**: ReadTheDocs can't access your GitHub repository.

**Solution**:

1. Check your GitHub integration:
   
   - Go to ReadTheDocs Admin > Integrations
   - Verify that the GitHub integration is connected

2. For private repositories:
   
   - Ensure ReadTheDocs has proper access to your repository
   - You may need to reconnect your GitHub account with appropriate permissions

Version Control Issues
-------------------

Missing Versions
~~~~~~~~~~~~~

**Problem**: Some versions of your documentation aren't available.

**Solution**:

1. Check your active versions:
   
   - Go to Admin > Versions
   - Activate the versions you want to build

2. Verify that tags and branches exist in your repository

3. Ensure your version names comply with ReadTheDocs naming conventions

Default Version Issues
~~~~~~~~~~~~~~~~~~~

**Problem**: The wrong version is shown as the default.

**Solution**:

1. Set your default version:
   
   - Go to Admin > Advanced Settings
   - Select your preferred default version

2. Make sure the selected version is active and builds successfully

Custom Domain Problems
-------------------

SSL Certificate Issues
~~~~~~~~~~~~~~~~~~~

**Problem**: SSL certificate errors with custom domain.

**Solution**:

1. Verify DNS configuration:
   
   - CNAME record should point to ``readthedocs.io``

2. Enable HTTPS in ReadTheDocs:
   
   - Go to Admin > Domains
   - Enable HTTPS for your domain

3. Wait for SSL certificate provisioning (can take up to 24 hours)

Domain Not Resolving
~~~~~~~~~~~~~~~~~

**Problem**: Custom domain doesn't resolve to your documentation.

**Solution**:

1. Check DNS configuration:
   
   - Verify CNAME record is correctly set up
   - Use DNS lookup tools to confirm propagation

2. Verify domain configuration in ReadTheDocs:
   
   - Go to Admin > Domains
   - Ensure domain is correctly configured

3. Wait for DNS propagation (can take up to 48 hours)

PDF/ePub Generation Issues
------------------------

PDF Build Failures
~~~~~~~~~~~~~~~

**Problem**: HTML documentation builds successfully, but PDF generation fails.

**Solution**:

1. Check for LaTeX errors in the build log

2. Simplify complex layouts:
   
   - Large tables
   - Complex diagrams
   - Custom directives

3. Install additional LaTeX packages if needed:

   .. code-block:: yaml

      # In .readthedocs.yml
      build:
        os: ubuntu-22.04
        tools:
          python: "3.10"
        apt_packages:
          - texlive-latex-extra
          - texlive-fonts-recommended

Missing Images in PDF
~~~~~~~~~~~~~~~~~~

**Problem**: Images appear in HTML but are missing in PDF.

**Solution**:

1. Ensure images are in a format supported by LaTeX (PNG, JPEG, PDF)

2. Check image paths:
   
   - Use relative paths
   - Avoid spaces in filenames

3. Add specific configurations for LaTeX output in ``conf.py``:

   .. code-block:: python

      latex_elements = {
          # latex options
      }

ePub Formatting Issues
~~~~~~~~~~~~~~~~~~~

**Problem**: ePub format has formatting or structure issues.

**Solution**:

1. Simplify complex layouts for better ePub compatibility

2. Check for ePub-specific warnings in the build log

3. Test the generated ePub with different readers

General Troubleshooting Tips
--------------------------

1. **Check Build Logs**: Always review the full build log for specific error messages.

2. **Local Testing**: Build documentation locally before pushing to identify issues early:

   .. code-block:: bash

      cd docs
      make html
      make latexpdf  # For PDF issues

3. **Incremental Fixes**: When troubleshooting complex issues, make small changes and test each change individually.

4. **ReadTheDocs Support**: For persistent issues, check the `ReadTheDocs documentation <https://docs.readthedocs.io/>`_ or contact their support team.

5. **Community Resources**: Search for similar issues in the ReadTheDocs or Sphinx communities:
   
   - `ReadTheDocs GitHub Issues <https://github.com/readthedocs/readthedocs.org/issues>`_
   - `Sphinx Users Mailing List <https://groups.google.com/forum/#!forum/sphinx-users>`_

By following this troubleshooting guide, you should be able to resolve most common issues encountered when deploying JAX-bandflux documentation to ReadTheDocs.