package dev.research.himanshu.ml.playground.imageclassification;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Map.Entry;

import javax.imageio.ImageIO;

/**
 * Author : Vibhav Gogate
 * 
 * The University of Texas at Dallas
 * 
 **/
public class KMeans {

	public static int iterations = 100;

	public static class ColorInstanceData {
		private double seperationDistance;
		private int cluster;

		public ColorInstanceData(double dist, int cluster) {
			this.seperationDistance = dist;
			this.cluster = cluster;
		}
	}

	public static void main(String[] args) {
		if (args.length < 3) {
			System.out.println("Usage: Kmeans <input-image> <k> <output-image>");
			System.out.println("Usage: Kmeans <input-image> <k> <output-image> <iterations>");
			return;
		}
		
		try {
			BufferedImage originalImage = ImageIO.read(new File(args[0]));
			int k = Integer.parseInt(args[1]);
			
			if (args.length == 4)
				iterations = Integer.parseInt(args[3]);

			BufferedImage kmeansJpg = kmeans_helper(originalImage, k);
			ImageIO.write(kmeansJpg, "jpg", new File(args[2]));

		} catch (IOException e) {
			System.out.println(e.getMessage());
		}
	}

	private static BufferedImage kmeans_helper(BufferedImage originalImage, int k) {

		int w = originalImage.getWidth();
		int h = originalImage.getHeight();

		BufferedImage kmeansImage = new BufferedImage(w, h, originalImage.getType());
		Graphics2D g = kmeansImage.createGraphics();

		g.drawImage(originalImage, 0, 0, w, h, null);

		// Read rgb values from the image
		int[] rgb = new int[w * h];
		int count = 0;
		for (int i = 0; i < w; i++) {
			for (int j = 0; j < h; j++) {
				rgb[count++] = kmeansImage.getRGB(i, j);
			}
		}

		// Call kmeans algorithm: update the rgb values
		kmeans(rgb, k);

		// Write the new rgb values to the image
		count = 0;
		for (int i = 0; i < w; i++) {
			for (int j = 0; j < h; j++) {
				kmeansImage.setRGB(i, j, rgb[count++]);
			}
		}

		return kmeansImage;
	}

	// Your k-means code goes here
	// Update the array rgb by assigning each entry in the rgb array to its
	// cluster center
	private static void kmeans(int[] rgb, int k) {
		
		// initialize the centroids
		int[] centroids = initializeCentroids(rgb, k);

		int[] prevCentroids = new int[centroids.length];
		System.arraycopy(centroids, 0, prevCentroids, 0, centroids.length);
		
		HashMap<Integer, Integer> rgbCountMap = new LinkedHashMap<Integer, Integer>();

		for (int i = 0; i < rgb.length; i++) {
			if (rgbCountMap.containsKey(rgb[i]))
				rgbCountMap.put(rgb[i], rgbCountMap.get(rgb[i]) + 1);
			else
				rgbCountMap.put(rgb[i], 1);
		}
		
		Map<Integer, ColorInstanceData> colorInstanceDataMap = new HashMap<Integer, ColorInstanceData>();

		boolean converged = false;

		// find the distance of all the pixels to the centroids selected
		int iteration = 1;
		while (!converged) {

			for (Map.Entry<Integer, Integer> rgbCountMapEntry : rgbCountMap.entrySet()) {
				double distance = distance(rgbCountMapEntry.getKey(), centroids[0]);
				
				colorInstanceDataMap.put(rgbCountMapEntry.getKey(), new ColorInstanceData(distance, 0));

				for (int j = 0; j < centroids.length; j++) {
					double newDist = distance (rgbCountMapEntry.getKey(), centroids[j]);
					
					if (newDist < colorInstanceDataMap.get(rgbCountMapEntry.getKey()).seperationDistance) {
						colorInstanceDataMap.get(rgbCountMapEntry.getKey()).seperationDistance = newDist;
						colorInstanceDataMap.get(rgbCountMapEntry.getKey()).cluster = j;
					}
				}
			}

			for (int j = 0; j < k; j++) {

				int[] bgrArr = new int[3];
				int c = 0;
				int m = 0;

				for (Entry<Integer, Integer> entry : rgbCountMap.entrySet()) {
					if (colorInstanceDataMap.get(entry.getKey()).cluster == j) {
						Color temp = new Color(entry.getKey());
						
						bgrArr[0] += (temp.getRed() * entry.getValue());
						bgrArr[1] += (temp.getBlue() * entry.getValue());
						bgrArr[2] += (temp.getGreen() * entry.getValue());
						
						c += entry.getValue();
					}
					m = m + 1;
				}

				if (c != 0) {
					Color temp = new Color(bgrArr[0] / c, bgrArr[1] / c, bgrArr[2] / c);
					centroids[j] = temp.getRGB();
				}
			}

			if (Arrays.equals(prevCentroids, centroids) || iteration == iterations) {
				System.out.println("convergence");
				converged = true;
				break;
			}

			System.arraycopy(centroids, 0, prevCentroids, 0, centroids.length);
			iteration++;
		}

		int p = 0;
		for (int i = 0; i < rgb.length; i++) {
			for (int j = 0; j < k; j++) {
				if (colorInstanceDataMap.get(rgb[i]).cluster == j) {
					rgb[i] = centroids[j];
					break;
				}
			}
			p = p + 1;
		}

	}

	/***
	 * function to find the distance between two pixels.
	 * 
	 * @param i
	 * @param j
	 * @return
	 */
	private static double distance(int i, int j) {
		Color point = new Color(i);
		Color centroid = new Color(j);

		double blue = Math.abs(Math.pow((point.getBlue() - centroid.getBlue()), 2));
		double red = Math.abs(Math.pow((point.getRed() - centroid.getRed()), 2));
		double green = Math.abs(Math.pow((point.getGreen() - centroid.getGreen()), 2));

		return Math.sqrt((blue + red + green));
	}

	/**
	 * function to initialize the centroids.
	 * 
	 * @param rgb
	 * @param k
	 * @return
	 */
	private static int[] initializeCentroids(int[] rgb, int k) {
		int centroids[] = new int[k];
		int length = rgb.length;
		
		for (int i = 0; i < k; i++) {
			int centroid = rgb [(int) (Math.floor (length * Math.random()))];
			boolean found = false;
		
			while (!found) {
				for (int j = 0; j < i - 1; j++) {
					if (centroids[j] == centroid) {
						centroid = rgb [(int) (Math.floor (length * Math.random()))];
						break;
					}
				}
				found = true;
			}
			centroids[i] = centroid;
		}
		
		return centroids;
	}

}