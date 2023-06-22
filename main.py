
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


if __name__ == '__main__':
    pic = mpimg.imread('./cat.png')
    plt.imshow(pic)
    plt.show()
    print(pic.shape)

    rows = pic.shape[0]
    cols = pic.shape[1]

    reshaped_image = pic.reshape(rows * cols, 3)
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(reshaped_image)
    print(kmeans.cluster_centers_)
    compressed_image = kmeans.cluster_centers_[kmeans.labels_]

    compressed_image = compressed_image.reshape(rows, cols, 3)
    plt.imshow(compressed_image)
    plt.imsave('./cat_compressed.png', compressed_image)
    plt.show()






