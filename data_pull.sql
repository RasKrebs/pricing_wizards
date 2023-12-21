SELECT DISTINCT
	c.id AS classified_id,
	DATE(c.created_at) AS listed_at_date,
	dp.id AS product_id,
	c.user_id,
	acp.price AS classified_price,
	retail_price,
	acp.listing_price,
	c.followers AS favourites,
	c.viewed_count,
	c.state,
	dcb.name AS brand_name,
	dcc.name AS condition_name,
	dcs.name AS size_name,
	dccol.name AS color_name,
	cat.category_name AS subcategory_name,
	cat.fourth_parent_name AS category_name
FROM
	classifieds c
	INNER JOIN auxiliary_tables.classifieds_dimensions cd ON cd.id = c.id
	LEFT JOIN dimensions.dim_products dp ON dp.category_id = cd.category_id
		AND dp.brand_id = cd.brand_id
		AND dp.condition_id = cd.condition_id
		AND dp.size_id = cd.size_id
		AND dp.color_id = cd.color_id
	LEFT JOIN dimensions.dim_classifieds_brands dcb ON dcb.id = dp.brand_id
	LEFT JOIN dimensions.dim_classified_conditions dcc ON dcc.id = dp.condition_id
	LEFT JOIN dimensions.dim_classified_sizes dcs ON dcs.id = dp.size_id
	LEFT JOIN dimensions.dim_classified_colors dccol ON dccol.id = dp.color_id
	LEFT JOIN dimensions.categories_by_parent cat ON cat.category = dp.category_id
	LEFT JOIN auxiliary_tables.auxiliary_classified_price acp ON acp.classified_id = c.id
WHERE
	c.created_at >= CURRENT_DATE - INTERVAL '3 months'
	AND c.state IN(5, 7, 8, 10);
