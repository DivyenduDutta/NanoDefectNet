class SingletonMeta(type):
    """
    A metaclass for creating singleton classes.

    It ensures that only one instance of a class is created,
    regardless of how many times the class is instantiated.
    """

    _instances: dict[type, object] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]
