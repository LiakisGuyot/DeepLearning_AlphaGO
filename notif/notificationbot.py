from pushover import init, Client


class NotificationBot:
    def __init__(self, user_key: str, api_token: str):
        self.user_key = user_key
        self.api_token = api_token
        init(api_token)

    def send_notification(self, title: str, message: str):
        """
        Send a notification to PushOver application linked.
        :param title:
        :param message:
        :return:
        """
        client = Client(self.user_key)
        client.send_message(message, title=title)
