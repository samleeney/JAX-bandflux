Comprehensive Guide to Deploying JAX-bandflux Documentation on ReadTheDocs
==================================================================

This guide provides step-by-step instructions for hosting and deploying the JAX-bandflux documentation on ReadTheDocs. It covers fixing common issues, setting up a ReadTheDocs account, configuring project settings, and maintaining documentation over time.

Table of Contents
----------------

1. `Fixing Documentation Issues`_
2. `Setting Up a ReadTheDocs Account`_
3. `Connecting to GitHub Repository`_
4. `Configuring ReadTheDocs Project Settings`_
5. `Triggering the Initial Build`_
6. `Setting Up Automatic Builds`_
7. `Best Practices for Documentation Maintenance`_
8. `Troubleshooting Common Issues`_

Fixing Documentation Issues
--------------------------

Before deploying to ReadTheDocs, ensure that your documentation is properly configured. Here are the common issues that need to be fixed:

Adding Missing Sphinx Extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you're using Mermaid diagrams in your documentation, you need to add the ``sphinxcontrib.mermaid`` extension to your ``conf.py`` file:

.. code-block:: python

    extensions = [
        'sphinx.ext.autodoc',
        'sphinx.ext.viewcode',
        'sphinx.ext.napoleon',
        'sphinx.ext.mathjax',
        'sphinx.ext.intersphinx',
        'sphinx.ext.autosummary',
        'sphinx_rtd_theme',
        'sphinxcontrib.mermaid',  # Add this line
    ]

Fixing RST Syntax Issues
~~~~~~~~~~~~~~~~~~~~~~~

In RST files, the underline for headings must match the length of the heading text. For example:

.. code-block:: rst

    # Incorrect
    Key Constants
    ------------

    # Correct
    Key Constants
    --------------

Check all your RST files for similar issues. Common RST syntax errors include:

- Inconsistent heading underline lengths
- Missing blank lines between sections
- Incorrect indentation in code blocks or lists
- Malformed links or references

Adding Documentation Dependencies to setup.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ensure that all documentation-related dependencies are included in your ``setup.py`` file under the 'dev' extras:

.. code-block:: python

    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=22.0',
            'isort>=5.0',
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=1.0.0',
            'sphinxcontrib-mermaid>=0.7.0',
            'sphinx-autodoc-typehints>=1.12.0',
        ],
    }

Verifying Your Documentation Locally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before deploying to ReadTheDocs, build your documentation locally to check for any issues:

.. code-block:: bash

    cd docs
    make html

Open ``_build/html/index.html`` in your browser to verify that the documentation builds correctly.

Setting Up a ReadTheDocs Account
-------------------------------

Create a ReadTheDocs Account
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Go to `ReadTheDocs.org <https://readthedocs.org/>`_ and click on "Sign Up"
2. You can sign up using your GitHub account for easier integration, or create a separate account
3. Complete the registration process and verify your email if required

Log In to Your Account
~~~~~~~~~~~~~~~~~~~~~

After creating your account, log in to the ReadTheDocs dashboard.

Connecting to GitHub Repository
-----------------------------

Import Your GitHub Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. In the ReadTheDocs dashboard, click on "Import a Project"
2. Connect your GitHub account if you haven't already
3. You'll see a list of your GitHub repositories. Find "JAX-bandflux" and click "Import"

Grant Repository Access
~~~~~~~~~~~~~~~~~~~~~

1. If prompted, grant ReadTheDocs access to your GitHub repository
2. For public repositories, you'll need to grant read access
3. For private repositories, you'll need to grant both read and webhook access

Initial Project Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

After importing the repository, you'll be taken to the project's admin page. Here you can configure basic settings:

1. Set the project name (default is the repository name)
2. Add a description for your project
3. Select the default branch to build documentation from (usually "main" or "master")
4. Choose the programming language (Python)

Configuring ReadTheDocs Project Settings
--------------------------------------

Advanced Project Settings
~~~~~~~~~~~~~~~~~~~~~~~

Navigate to the "Admin" tab of your project and configure the following settings:

1. **Documentation Type**: Set to "Sphinx"
2. **Programming Language**: Set to "Python"
3. **Python Configuration File**: Set to "docs/conf.py"
4. **Python Interpreter**: Set to "CPython 3.x" (preferably 3.10 as specified in your .readthedocs.yml)

Build Settings
~~~~~~~~~~~~

Under the "Advanced Settings" section:

1. **Install Project**: Enable this option to install your package during the build
2. **Requirements File**: Leave empty as we're using the .readthedocs.yml configuration
3. **Python Installation**: Set to "pip"
4. **Use System Packages**: Disable this option

Versions
~~~~~~~

Under the "Versions" tab:

1. Activate the branches you want to build documentation for
2. Set your default version (usually "latest" which corresponds to your default branch)

Integrations
~~~~~~~~~~

Under the "Integrations" tab:

1. Verify that the GitHub integration is active
2. Check that the webhook has been properly set up

Triggering the Initial Build
--------------------------

Manual Build
~~~~~~~~~~

1. Go to the "Builds" tab in your project dashboard
2. Click on "Build Version" for your default version
3. Monitor the build process for any errors

Verifying the Build
~~~~~~~~~~~~~~~~~

1. Once the build completes successfully, click on "View Docs" to see your published documentation
2. Check that all pages render correctly
3. Verify that navigation works as expected
4. Ensure that all images, diagrams, and code examples display properly

Troubleshooting Build Failures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your build fails:

1. Click on the failed build to view the build log
2. Look for error messages that indicate what went wrong
3. Common issues include:
   - Missing dependencies
   - Syntax errors in RST files
   - Configuration errors in conf.py
   - Import errors when autodoc tries to import your modules

Setting Up Automatic Builds
-------------------------

ReadTheDocs automatically sets up a webhook in your GitHub repository to trigger builds when you push changes.

Verifying Webhook Setup
~~~~~~~~~~~~~~~~~~~~~

1. Go to your GitHub repository
2. Navigate to Settings > Webhooks
3. Verify that a ReadTheDocs webhook is listed and active

Testing Automatic Builds
~~~~~~~~~~~~~~~~~~~~~~

1. Make a small change to your documentation
2. Commit and push the change to your repository
3. Go to the "Builds" tab in ReadTheDocs to verify that a new build was triggered

Build Notifications
~~~~~~~~~~~~~~~~

Configure build notifications to be alerted of build successes or failures:

1. Go to your ReadTheDocs account settings
2. Under "Email", configure your notification preferences
3. You can choose to receive emails for successful builds, failed builds, or both

Best Practices for Documentation Maintenance
-----------------------------------------

Documentation Structure
~~~~~~~~~~~~~~~~~~~~~

Maintain a clear and consistent structure:

- **Installation Guide**: Keep installation instructions up-to-date
- **Quick Start Guide**: Provide simple examples to get users started
- **Tutorials**: Step-by-step guides for common tasks
- **API Reference**: Comprehensive documentation of all modules, classes, and functions
- **Examples**: Real-world usage examples

Version Control
~~~~~~~~~~~~~

- Tag releases in your repository to create versioned documentation
- In ReadTheDocs, activate important release tags to make them available in the version selector

Documentation Testing
~~~~~~~~~~~~~~~~~~~

- Regularly build documentation locally to catch issues early
- Consider setting up documentation testing in your CI pipeline
- Use tools like ``doc8`` to check RST syntax

Keep Dependencies Updated
~~~~~~~~~~~~~~~~~~~~~~~

- Regularly update Sphinx and its extensions
- Update your theme and custom extensions
- Test with newer Python versions

Improve Accessibility and Usability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Use descriptive link text
- Provide alt text for images
- Ensure good color contrast
- Use consistent heading levels

Documentation Reviews
~~~~~~~~~~~~~~~~~~~

- Include documentation updates in code reviews
- Have team members review documentation changes
- Collect user feedback on documentation clarity

Troubleshooting Common Issues
---------------------------

Build Failures
~~~~~~~~~~~~

1. **Missing Dependencies**:
   - Ensure all dependencies are listed in setup.py or requirements.txt
   - Check that the .readthedocs.yml file correctly specifies how to install dependencies

2. **Import Errors**:
   - Make sure your package is installed during the documentation build
   - Check that any C extensions or compiled components can be built on ReadTheDocs

3. **Sphinx Extension Issues**:
   - Verify that all required Sphinx extensions are installed
   - Check for compatibility issues between extension versions

Content Issues
~~~~~~~~~~~~

1. **Broken Links**:
   - Use tools like ``sphinx-build -b linkcheck`` to find broken links
   - Regularly check external links

2. **Missing or Outdated Content**:
   - Set up a regular schedule to review and update documentation
   - Add documentation updates to your release checklist

3. **Rendering Problems**:
   - Test documentation with different browsers
   - Check mobile responsiveness

ReadTheDocs Specific Issues
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Webhook Not Triggering**:
   - Manually check webhook delivery in GitHub
   - Reconfigure the webhook if necessary

2. **Custom Domain Issues**:
   - Follow the ReadTheDocs guide for setting up custom domains
   - Ensure DNS settings are correct

3. **PDF/ePub Generation Issues**:
   - Check for LaTeX dependencies if building PDF documentation
   - Simplify complex layouts that might cause PDF generation to fail

----

By following this guide, you should be able to successfully deploy and maintain your JAX-bandflux documentation on ReadTheDocs. Remember to regularly update your documentation as your project evolves to ensure it remains a valuable resource for your users.