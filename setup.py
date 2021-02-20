from setuptools import setup, find_packages


def setup_package():
    setup(
        name='UIUC_PSL',
        packages=find_packages(exclude=('test',)),
        description='',
        author='Fan Yang',
        author_email='yfno1@msn.com',
        scripts=[],
        zip_safe=False,
        include_package_data=True
    )


if __name__ == "__main__":
    setup_package()
