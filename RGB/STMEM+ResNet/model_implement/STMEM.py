import torch
import torch.nn as nn

class STMEM(nn.Module):
    def __init__(self, num_segments, new_length, img_size=(224, 224)):
        super(STMEM, self).__init__()
        self.num_segments = num_segments
        self.new_length = new_length
        self.height, self.width = img_size

        self.sigmoid = nn.Sigmoid()

        self.m1 = nn.Sequential(
            nn.Conv2d(
                in_channels=(self.new_length * 2 - 1) * 3,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

        self.m2 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

    def forward(self, x):
        # Input shape: (B, S * L * 3, H, W)
        B, SLC, H, W = x.size()
        assert H == self.height and W == self.width, \
            f"Expected input height={self.height}, width={self.width} but got {H}x{W}"

        # Reshape to: (B * S, L * 3, H, W)
        x = x.view(B * self.num_segments, self.new_length * 3, self.height, self.width)

        # Compute frame differences (temporal modeling)
        frame_diff = x[:, 3:] - x[:, : (self.new_length - 1) * 3]

        # Concatenate original input with motion information
        x_with_diff = torch.cat((x, frame_diff), dim=1)
        x_with_diff = self.m1(x_with_diff)

        # Get max motion frame
        frame_diff = frame_diff.view(B * self.num_segments, self.new_length - 1, 3, self.height, self.width)
        frame_diff = frame_diff.max(dim=1)[0]

        frame_diff = self.m2(frame_diff)
        motion_mask = self.sigmoid(frame_diff)

        # Apply motion mask
        output = motion_mask * x_with_diff

        return output  # shape: (B * S, 3, H, W)

if __name__ == '__main__':
    a = torch.rand([4,90,224,224])
    model = STMEM(num_segments=5,new_length=6)
    out = model(a)
    print(out.size())